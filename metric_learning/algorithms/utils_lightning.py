#!/usr/bin/env python3
from typing import Any, Optional
import torch
from torch import optim
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.nn.modules.loss import CrossEntropyLoss

import numpy as np
from metric_learning.arguments import PreTrainArguments, FewshotArguments
from metric_learning.algorithms.metric_learner import MetricLearner


def default_logger_step(
    logger: Any, prelabel: str, loss: Optional[float], accuracy: Optional[float]
):
    if loss is not None:
        logger(
            f"{prelabel}-loss",
            loss,
            on_step=False,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

    if accuracy is not None:
        logger(
            f"{prelabel}-accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

class LightningPretrainModule(pl.LightningModule):
    def __init__(
        self,
        metric_model: MetricLearner,
        training_arguments: PreTrainArguments,
        **kwargs,
    ) -> None:
        super().__init__()
        self.metric_model = metric_model
        self.training_arguments = training_arguments
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.metric_model.init_pretraining(**kwargs)
        # self.save_hyperparameters()

    def training_step(self, batch: torch.Tensor, _) -> torch.Tensor:
        inputs, labels = batch
        logits = self.metric_model(inputs)
        labels = labels.long()
        loss = self.loss_fn(logits, labels)
        acc = accuracy(logits.softmax(dim=-1), labels)
        default_logger_step(self.log, "training", loss.item(), float(acc.item()))
        return loss

    def validation_step(self, batch: torch.Tensor, _) -> Optional[Any]:
        inputs, labels = batch
        logits = self.metric_model(inputs)
        labels = labels.long()
        loss = self.loss_fn(logits, labels)
        acc = accuracy(logits.softmax(dim=-1), labels)
        default_logger_step(self.log, "validation", loss.item(), float(acc.item()))
        return loss.item()

    def test_step(self, batch: torch.Tensor, _) -> Optional[Any]:
        inputs, labels = batch
        labels = labels.long()
        logits = self.metric_model(inputs)
        loss = self.loss_fn(logits, labels)
        acc = accuracy(logits.softmax(dim=-1), labels)
        default_logger_step(self.log, "test", loss.item(), float(acc.item()))
        return loss.item()

    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler._LRScheduler]]:
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.training_arguments.lr,
            momentum=self.training_arguments.momentum,
            weight_decay=self.training_arguments.weight_decay
        )
        lr_scheduler: optim.lr_scheduler._LRScheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1.0
        )

        if self.training_arguments.milestones is not None:
            print(f"{self.training_arguments=}")
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.training_arguments.milestones,
                gamma=self.training_arguments.lr_reduce,
            )

        return [optimizer], [lr_scheduler]


class LightningFewshotModule(pl.LightningModule):
    """Lightning Module for training few shot learners
    """
    def __init__(self, learner: MetricLearner, fs_arguments: FewshotArguments) -> None:
        super().__init__()
        self.learner = learner
        self.fs_arguments = fs_arguments
        self.loss = CrossEntropyLoss()
        # self.save_hyperparameters()

    def _support_query_split(
            self, batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        """

        images, labels = batch

        assert images.size(0) == self.fs_arguments.ways * (
            self.fs_arguments.shots + self.fs_arguments.queries
        )

        support_indices = np.zeros(images.size(0), dtype=bool)
        selection = np.arange(self.fs_arguments.ways) * (
            self.fs_arguments.shots + self.fs_arguments.queries
        )
        for offset in range(self.fs_arguments.shots):
            support_indices[selection + offset] = True

        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)

        support = images[support_indices]
        support_labels = labels[support_indices]
        query = images[query_indices]
        query_labels = labels[query_indices]

        # Reshape
        # support = support.reshape(
        #     self.fs_arguments.ways, self.fs_arguments.shots, *support.shape[1:]
        # )
        # query_labels = torch.tensor(np.repeat(list(range(self.fs_arguments.ways)),self.fs_arguments.queries))
        return support, support_labels, query, query_labels

    def _meta_step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        support, support_labels, query, query_labels = self._support_query_split(batch)
        logits = self.learner(query, support, support_labels)
        loss = self.loss(logits, query_labels)
        acc = accuracy(logits.softmax(dim=-1), query_labels)
        return loss, acc

    def training_step(self, batch, _) -> torch.Tensor:
        loss, acc = self._meta_step(batch)
        default_logger_step(self.log, "training", float(loss.item()), float(acc.item()))
        return loss

    def validation_step(self, batch, _) -> torch.Tensor:
        loss, acc = self._meta_step(batch)
        default_logger_step(self.log, "validation", float(loss.item()), float(acc.item()))
        return loss

    def test_step(self, batch, _) -> torch.Tensor:
        loss, acc = self._meta_step(batch)
        default_logger_step(self.log, "test", float(loss.item()), float(acc.item()))
        return loss

    def configure_optimizers(self) -> list[optim.Optimizer]:
        optimizer = optim.SGD(self.parameters(), lr=self.fs_arguments.lr)
        return [optimizer]
