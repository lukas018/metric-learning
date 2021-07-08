#!/usr/bin/env python3
import torch
import abc
from typing import Any


class MetricLearner(torch.nn.Module):

    @abc.abstractmethod
    def init_pretraining(self, *args, **kwargs) -> Any:
        pass

    def init_fewshot(self, *args, **kwargs): -> Any:
        pass
