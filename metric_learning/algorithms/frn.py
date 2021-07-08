from functools import cache
from itertools import permutations
from typing import Callable, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from metric_learning.algorithms.metric_learner import MetricLearner
from torch import nn


@cache
def non_overlapping_pair_indices(n: int):
        i1, i2 = (
            *map(
                torch.tensor,
                zip(*permutations(list(range(n)), 2)),
            ),
        )
        return i1, i2


class FeatureReconNetwork(MetricLearner):
    """Feature Reconstruction Network"""

    def __init__(
        self,
        model: nn.Module,
        nchannels: int,
        resolution: int,
        alpha: float = 0.0,
        beta: float = 0.0,
        scale_factor: float = 1 / np.sqrt(640),
        aux_loss_scale: float = 0.03,
        temperature: float = 1.0,
        loss_fn: Optional[Callable] = None,
        woodbury: bool = True,
    ):
        """Feature Reconstruction Network

        :param model: The feature extractor (any cnn of equivelent)
        :param nchannels: The number of output channels on the final layer of
            model
        :param resolution: Height x width of the feature maps outputed by model
        :param alpha: Initial value for learnable parameter alpha
        :param beta: Initial value for learnable parameter beta
        :param scale_factor:
        :param aux_loss_scale:
        :param temperature:
        :param loss_fn:
        :param woodbury: Whether to use woodbury identity during feature reconstruction.
        Recommended for larger k's
        """

        super().__init__()
        self.model = model

        # This corresponds to d in the paper
        self.nchannels = nchannels
        # This corresponds to r in the paper
        self.resolution = resolution

        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.class_matrices: Optional[torch.Tensor] = None
        self.cached_support: Optional[torch.Tensor] = None

        self.scale_factor = scale_factor
        self.aux_loss_scale = aux_loss_scale

        self.criterion = nn.NLLLoss() if loss_fn is None else loss_fn
        self.woodbury = woodbury

    def init_pretraining(self, num_classes: int):
        """Initialize the learner for pre-training

        :param num_classes: The number of classes ~n~ in the pretraining dataset
        """

        if self.class_matrices is None or self.class_matrices.shape[0] == num_classes:
            self.class_matrices = nn.Parameter(torch.randn(
                (num_classes, self.resolution, self.nchannels),
            ))

    def compute_support(
        self,
        support: torch.Tensor,
        cache: bool = False,
    ) -> torch.Tensor:
        """Compute and return the class representations based on the given
        support set.

        :param support: The set of support images in format
        :param cache: Set to true to save the support representation.  This will
            make the model perform fewshot classification during the forward
            pass using the currently specified classes even if no support images
            are provided.
        """
        # Do few-shot prediction
        nway = support.shape[0]

        # [nway*show, channels, ...]
        features = self.model(support.flatten(0,1))

        # nway*shots, resolution, channels
        features = self.format_features(features)

        # nway, shot*rsolution, channels
        features = features.reshape(nway, -1, self.nchannels)
        features *= self.scale_factor

        if cache:
            self.cached_support = nn.Parameter(features)

        return features

    def format_features(self, features):
        bsz = features.shape[0]

        # Remember that its channel first in PyTorch
        return features.view(bsz, self.nchannels, -1).permute(0,2,1)

    def format_support(self, support, support_labels=None):

        if support_labels is None:
            if len(support.shape) > 4:
                raise TypeError(
                    f"Missmatched shape of support. If no support_labels are provided support needs to be shaped as [n, k, ...image_size]"
                )
            return support
        else:
            unique_support_labels, support_labels_count = support_labels.unique(
                dim=0, return_counts=True
            )

            # FRN Requires that k is equal for all
            assert ((support_labels_count - support_labels_count[0]) == 0).sum()

            # Sort the images based on label
            idx = torch.argsort(support_labels)
            support = support[idx, :]
            support = support.reshape((
                len(unique_support_labels),
                support_labels_count[0],
                *support.shape[1:],),
            )
            return support

    def forward(
        self,
        query: torch.Tensor,
        support: Optional[torch.Tensor] = None,
        support_labels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
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
        :param support: Set of images such that shape=((n*k) x h x w x channels)
        :param support_labels: Labels for the
        :param labels: Labels for the fewshot classification task

        :raises ValueError: If no support images are provided and the model
            neither have a cached set of class representations nor have called
            init_pretraining

        :returns: Prediction for each query image.  Logit dimension depends on
                 prediction mode-used.
        """

        # Compute and flatten the input features
        # [bsz, resolution, channels]

        query = self.format_features(self.model(query)) * self.scale_factor

        if support is not None or self.cached_support is not None:

            if support is not None:
                # If support is flat, convert it to the standard structure
                support = self.format_support(support, support_labels=support_labels)
                support = self.compute_support(support)
            else:
                support = self.cached_support

            assert support is not None
            # nway = support.shape[0]

            recons = self._reconstruct(query, support, self.alpha, self.beta)

            # ways, bsz*resolution, nchannels
            aux_loss = self._aux_loss(recons)

            logits = self._predictions(recons, query)
            return_struct = (logits,)

            if labels is not None:
                loss = self.criterion(logits, labels)
                loss += aux_loss
                return_struct += (loss,)

            return return_struct
        else:
            # Standard predictions
            if self.class_matrices is None:
                raise ValueError(
                    "Class matrices were not initialized, please run init_pretraining before calling method without support data",
                )
            ones = torch.ones(
                1,
                dtype=self.class_matrices.dtype,
                device=self.class_matrices.device
            )
            recons = self._reconstruct(query, self.class_matrices, ones, ones)
            logits = self._predictions(recons, query)
            return_struct = (logits,)
            if labels is not None:
                loss = self.criterion(logits, labels)
                return_struct += (loss,)

            return return_struct

    def _reconstruct(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature reconstructions of the query images

        :param query: [bsz, r, d]
        :param support: [way, support_shot* r, d]
        :param alpha:
        :param beta:
        """

        # this code snippet is copied more or less straight from the paper's appendix and Equation 8

        # Flatten everything
        query = query.flatten(0, 1)
        lam = support.shape[1] / support.shape[2] * alpha.exp()
        rho = beta.exp()
        st = support.permute(0, 2, 1)

        if self.woodbury:
            xtx = st @ support
            eye = torch.eye(xtx.shape[-1], dtype=query.dtype, device=query.device).unsqueeze(0)
            m_inv = (xtx + eye * lam).inverse()
            hat = m_inv @ xtx
        else:
            xtx = st @ support
            eye = torch.eye(xtx.shape[-1], dtype=query.dtype, device=query.device).unsqueeze(0)
            m_inv = (xtx + eye * lam).inverse()
            hat = m_inv @ xtx

        # way, bsz*resoultion, nchannels
        Q_bar = (query @ hat) * rho
        return Q_bar

    def _predictions(self, recons, query) -> torch.Tensor:
        """Computes the (normalized) logits from the reconstruction and original features

        :param recons: The reconstructed features of the query set, shape [way, bsz*resolution, channels]
        :param query: The query input images, [bsz, resolution, num_channels]
        """

        # [1, bsz*resolution, num_channels]
        bsz = query.shape[0]
        query = query.flatten(0, 1).unsqueeze(0)


        # [way, bsz*resolution, num_channels]
        dists = recons - query

        # [bsz*resolution, way]
        dists = dists.pow(2).sum(2).permute(1, 0)

        # [bsz, way]
        dists = dists.reshape((bsz, self.resolution, -1)).mean(1)
        dists *= -self.temperature
        return F.log_softmax(dists, dim=-1)

    def _aux_loss(self, recons) -> torch.Tensor:
        """Auxiluary loss

        :param recons: reconstruction of the classes [nway, bsz*resolutions, channels]
        """

        n, batch_resolution = recons.shape[:2]

        # Row normalize
        recons = recons / recons.norm(2)

        # Compute non ovelapping indicies pairs
        i1, i2 = non_overlapping_pair_indices(n)
        dists = recons.index_select(0, i1) @ recons.index_select(0, i2).permute(0, 2, 1)
        assert dists.shape[-1] == batch_resolution
        return dists.pow(2).sum(-1).sum(-1).sum() * self.aux_loss_scale



