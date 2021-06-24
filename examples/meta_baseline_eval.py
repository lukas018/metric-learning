#!/usr/bin/env python3
import pytorch_lightning as pl
from metric_learning.arguments import FewshotArguments
import torch
from metric_learning.data import EpisodicBatcher, initialize_mini_imagenet
from metric_learning.algorithms.meta_baseline import MetaBaseline
from metric_learning.algorithms.utils_lightning import (
    LightningFewshotModule,
)
from metric_learning.arguments import FewshotArguments

model: MetaBaseline = torch.load("meta-baseline-pretraining/pretrained.pth")
fs_args = FewshotArguments(
    ways=5,
    shots=1,
    gpus=[0],
    lr=0.001,
    meta_batch_size=4,
    batch_per_epoch=50,
    n_epochs=1,
    n_test_tasks=100,
)

datasets, metadatasets, task_datasets = initialize_mini_imagenet(fs_args)
datasets[0].transform = datasets[1].transform
meta_trainer = pl.Trainer(
    gpus=fs_args.gpus,
    weights_save_path=fs_args.checkpoint,
    accumulate_grad_batches=fs_args.meta_batch_size,
    max_epochs=fs_args.n_epochs,
)


fs_module = LightningFewshotModule(model, fs_args)
episodic_batcher = EpisodicBatcher(*task_datasets, epoch_length=fs_args.epoch_length)
meta_trainer.test(fs_module, episodic_batcher.test_dataloader())
print(model.temperature)
import torch.nn as nn
model.temperature = nn.Parameter(torch.tensor(10.))
for _ in range(10):
    meta_trainer.fit(fs_module, episodic_batcher)
    meta_trainer.test(fs_module, episodic_batcher.test_dataloader())
    print(model.temperature)
