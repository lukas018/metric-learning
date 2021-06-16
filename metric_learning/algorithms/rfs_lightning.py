#!/usr/bin/env python3
import copy
from metric_learning.arguments import PreTrainArguments
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from .utils_lightning import default_logger_step
from dataclasses import dataclass, field


class DistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, temperature: float):
        super(DistillKL, self).__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits) -> torch.Tensor:
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        loss = F.kl_div(student_probs, teacher_probs, size_average=False)
        loss *= (self.temperature**2) / student_logits.shape[0]
        return loss


@dataclass
class DistillArguments(PreTrainArguments):
    n_generations: int = field(default=2)
    alpha: float = field(default=1.0)
    beta: float = field(default=0.0)
    gamma: float = field(default=1.0)


class BornAgainLightningModule(pl.LightningModule):
    """Perform distillation"""

    def __init__(self, teacher, training_arguments, gamma, alpha, beta, student=None, temperature=4.0):
        super().__init__()
        self.training_arguments = training_arguments
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cls_loss = nn.CrossEntropyLoss()
        self.difference_loss = DistillKL(temperature)

        self.teacher = teacher
        self.student = student

        # Try to reinitiailize the student if we dont
        if student is None:
            self.student = copy.deepcopy(self.teacher)

            # Reset the parameters
            def weight_reset(m):
                reset_parameters = getattr(m, "reset_parameters", None)
                if callable(reset_parameters):
                    m.reset_parameters()

            self.student.apply(weight_reset)
        # self.save_hyperparameters()

    def _loss(self, student_logits, teacher_logits, labels):
        cls_loss = self.cls_loss(student_logits, labels.long())
        diff_loss = self.difference_loss(student_logits, teacher_logits)
        loss = (self.alpha * cls_loss) + (self.beta * diff_loss)
        return loss

    def training_step(self, batch, _) -> torch.Tensor:
        images, labels = batch

        student_logits = self.student(images)

        with torch.no_grad():
            teacher_logits = self.teacher(images)

        loss = self._loss(student_logits, teacher_logits, labels)
        acc = accuracy(student_logits.softmax(dim=-1), labels.long())
        default_logger_step(self.log, "training", loss, acc.item())
        return loss

    def validation_step(self, batch, _):
        images, labels = batch
        logits = self.student(images)
        acc = accuracy(logits.softmax(dim=-1), labels.long())
        default_logger_step(self.log, "validation", None, acc.item())

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
                gamma=self.training_arguments.lr_reduce
            )

        return [optimizer], [lr_scheduler]
