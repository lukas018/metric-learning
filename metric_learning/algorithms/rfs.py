#!/usr/bin/env python3
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cdist
from torch import linalg as LA
from torch import nn

from enum import Enum
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metric_learner import MetricLearner


class RFSClassifier(Enum):
    LogisticRegression = 0
    SVM = 1


def build_classifier(
    support, support_labels, cls_type: RFSClassifier, **kwargs
) -> Callable:
    support = support.detach().cpu().numpy()
    support_labels = support_labels.detach().cpu().numpy()

    if cls_type == RFSClassifier.LogisticRegression:
        # Extract the numpy from
        classifier = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            max_iter=1000,
            multi_class="multinomial",
        )
        classifier.fit(support, support_labels)
        return lambda xs: classifier.predict_proba(xs)
    elif cls_type == RFSClassifier.SVM:
        classifier = make_pipeline(
            StandardScaler(),
            SVC(gamma="auto", C=1, kernel="linear", decision_function_shape="ovr"),
        )
        classifier.fit(support, support_labels)
        return lambda xs: classifier.predict(xs)


class RepresentationForFS(MetricLearner):
    """Representation for Few-shot Learning"""

    def __init__(self, model: nn.Module, classfier_type: RFSClassifier):
        """Feature Reconstruction Network

        :param model: The feature extractor (any cnn or equivelent)
        """
        super().__init__()
        self.model = model
        self.classifier_type = classfier_type
        self.class_matrix = None
        self.classifier = None

    def init_pretraining(self, dimensions: int, num_classes: int):
        """Initialize the learner for pre-training

        :param num_classes: The number of classes ~n~ in the pretraining dataset
        """
        if self.class_matrix is None or self.class_matrix.out_features == num_classes:
            self.class_matrix = nn.Linear(dimensions, num_classes)

    def build_classifier(self, support: torch.Tensor, support_labels: torch.Tensor , cache: bool = False):
        classifier = build_classifier(support, support_labels, self.classifier_type)
        if cache:
            self.classifier = classifier
        return classifier

    def forward(
        self,
        query: torch.Tensor,
        support: Optional[torch.Tensor] = None,
        support_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict labels using FRN.

        This function offerst two different modes: standard prediction and
        few-shot predictions.  Standard prediction acts standard image
        classification with logits of n-classes.  This mode is enabled by
        default and is meant to be used during the pre-training phase outlined
        in the original paper.

        Few-shot prediction is used when support-images are given as arguments
        or compute_support has been called with cache set to True.  Support
        images are a set of n x k images (n classes, with k examples in each)
        and is used to compute the class representation matrices used for
        prediction.

        :param query: Set of images such that shape=(bsz x h x w x channels)
        :param support: Set of images such that shape=(n x k x h x w x channels)

        :raises ValueError: If no support images are provided and the model
            neither have a cached set of class representations nor have called
            init_pretraining

        :returns: Prediction for each query image.  Logit dimension depends on
                 prediction mode-used.
        """

        if support is not None or self.classifier is not None:
            with torch.no_grad():
                support_features = self.model(support)
                if support is not None:

                    if support_labels is None:
                        raise ValueError("No labels provided")

                    classifier = self.build_classifier(support_features, support_labels)
                else:
                    classifier = self.classifier

                query_features = self.model(query)
                logits = classifier(query_features.detach().cpu().numpy())
                logits = torch.Tensor( logits ).type_as(query)

        else:
            # Standard predictions
            if self.class_matrix is None:
                raise ValueError(
                    "Class matricx were not initialized, please run init_pretraining before calling method without support data",
                )
            features = self.model(query)
            logits = self.class_matrix(features)

        return logits
