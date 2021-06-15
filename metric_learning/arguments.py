#!/usr/bin/env python3
from typing import Optional, List
from dataclasses import dataclass, field

@dataclass
class PreTrainArguments:
    lr: float = 0.1

    milestones: List[int] = field(
        default_factory=lambda: [40, 80], metadata={"help": "Epochs at which to reduce the learning rate"}
    )
    gamma: float = field(
        default=0.1, metadata={"help": "Rate to reduce the learning rate with"}
    )
    n_epochs: int = field(default=100, metadata={"help": "Number of epochs to train"})
    checkpoint: Optional[str] = field(
        default="checkpoints", metadata={'help': "Checkpoint directory"},
    )
    log: Optional[str] = field(
        default="logs", metadata={'help': "Checkpoint directory"},
    )
    batch_size: int = field(
        default=64, metadata={"help": "Size of gradient batch"}
    )
    accumulate_grad_batches: int = field(
        default=4, metadata={"help": "Number of actual batches"}
    )



@dataclass
class FewshotArguments:
    lr: float = 0.001
    ways: int = 5
    shots: int = 1
    queries: int = 15

    checkpoint: Optional[str] = None
    log: Optional[str] = None

    n_epochs: int = field(default=100, metadata={"help": "Number of epochs to train"})
    meta_batch_size: int = field(default=16, metadata={"help": "Number of epochs to train"})
    batch_per_epoch: int = field(default=20, metadata={"help": "Number of meta-batches per epoch"})

    def __post_init__(self):
        self.epoch_length = self.meta_batch_size * self.batch_per_epoch
