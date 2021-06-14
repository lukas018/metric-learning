#!/usr/bin/env python3

from typing import Any, Optional
import torch
from torch import optim
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.nn.modules.loss import CrossEntropyLoss

from dataclasses import dataclass, field
import numpy as np

from ..metric_learner import MetricLearner


@dataclass
class PreTrainArguments:
    lr: float = 0.1
    milestones: Optional[list[int]] = field(
        default=None, metadata={"help": "Epochs at which to reduce the learning rate"}
    )
    gamma: float = field(
        default=0.1, metadata={"help": "Rate to reduce the learning rate with"}
    )
    n_epochs: int = field(default=100, metadata={"help": "Number of epochs to train"})
    checkpoint: Optional[str] = None
    log: Optional[str] = None


@dataclass
class FewshotArguments:
    lr: float = 0.001
    ways: int = 5
    shots: int = 5
    queries: int = 15

    checkpoint: Optional[str] = None
    log: Optional[str] = None

    n_epochs: int = field(default=100, metadata={"help": "Number of epochs to train"})
    meta_batch_size: int = field(default=16, metadata={"help": "Number of epochs to train"})
    batch_per_epoch: int = field(default=100, metadata={"help": "Number of meta-batches per epoch"})

    def __post_init__(self):
        self.epoch_length = self.meta_batch_size * self.batch_per_epoch


def _log_step(logger, prelabel: str, loss: float, accuracy: float):
    logger(
        f"{prelabel}-loss",
        loss,
        on_step=False,
        on_epoch=False,
        prog_bar=False,
        logger=True,
    )
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
        self.metric_model = metric_model
        self.training_arguments = training_arguments
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.metric_model.init_pretraining(**kwargs)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        inputs, labels = batch
        logits = self.metric_model(inputs)
        loss = self.loss_fn(logits, labels)
        acc = accuracy(logits, labels)
        _log_step(self.log, "training", loss.item(), float(acc.item()))
        return loss

    def validation_step(self, batch: torch.Tensor) -> Optional[Any]:
        inputs, labels = batch
        logits = self.metric_model(inputs)
        loss = self.loss_fn(logits, labels)
        acc = accuracy(logits, labels)
        _log_step(self.log, "validation", loss.item(), float(acc.item()))
        return loss.item()

    def test_step(self, batch: torch.Tensor) -> Optional[Any]:
        inputs, labels = batch
        logits = self.metric_model(inputs)
        loss = self.loss_fn(logits, labels)
        acc = accuracy(logits, labels)
        _log_step(self.log, "test", loss.item(), float(acc.item()))
        return loss.item()

    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler._LRScheduler]]:
        optimizer = optim.SGD(self.parameters(), lr=self.training_arguments.lr)
        lr_scheduler: optim.lr_scheduler._LRScheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1.0
        )

        if self.training_arguments.milestones is not None:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.training_arguments.milestones,
                gamma=self.training_arguments.gamma,
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
        selection = np.arange(self.fs_arguments.ways) * ()
        for offset in range(self.fs_arguments.shots):
            support_indices[selection + offset] = True

        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)

        support = images[support_indices]
        support_labels = labels[support_indices]
        query = images[query_indices]
        query_labels = labels[query_indices]
        return support, support_labels, query, query_labels

    def _meta_step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        support, _, query, query_labels = self._support_query_split(batch)
        logits = self.learner(support, query)
        loss = self.loss(logits, query_labels)
        acc = accuracy(logits, query_labels)
        return loss, acc

    def training_step(self, batch) -> torch.Tensor:
        loss, acc = self._meta_step(batch)
        _log_step(self.log, "training", float(loss.item()), float(acc.item()))
        return loss

    def validation_step(self, batch) -> torch.Tensor:
        loss, acc = self._meta_step(batch)
        _log_step(self.log, "validation", float(loss.item()), float(acc.item()))
        return loss

    def test_step(self, batch) -> torch.Tensor:
        loss, acc = self._meta_step(batch)
        _log_step(self.log, "test", float(loss.item()), float(acc.item()))
        return loss

    def configure_optimizers(self) -> list[optim.Optimizer]:
        optimizer = optim.SGD(self.parameters(), lr=self.fs_arguments.lr)
        return [optimizer]
