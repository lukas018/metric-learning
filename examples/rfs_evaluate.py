#!/usr/bin/env python3
import torch
from metric_learning.arguments import FewshotArguments
import pytorch_lightning as pl
from metric_learning.algorithms.utils_lightning import (
    LightningFewshotModule,
)

from metric_learning.data import EpisodicBatcher, initialize_mini_imagenet
from metric_learning.algorithms.rfs import RepresentationForFS, RFSClassifier
from metric_learning.algorithms.meta_baseline import MetaBaseline
from metric_learning.models.resnet import resnet12


fs_args = FewshotArguments(
    ways=5,
    shots=1,
    batch_per_epoch=100,
    meta_batch_size=1
)

model: MetaBaseline = torch.load("meta-baseline-pretraining/pretrained.pth")
feat_extractor = model.model
rfs = RepresentationForFS(feat_extractor, RFSClassifier.LogisticRegression)

datasets, metadatasets, task_datasets = initialize_mini_imagenet(fs_args)
episodic_batcher = EpisodicBatcher(*task_datasets, epoch_length=fs_args.epoch_length)

# feat_extractor = resnet12(avg_pool=True)
# rfs = torch.load("rfs-models-distill/rfs-distilled-model-org")
fs_module = LightningFewshotModule(rfs, fs_arguments=fs_args)
fs_trainer = pl.Trainer(
    gpus=1,
    default_root_dir=fs_args.checkpoint
)
# We don't need to train here
fs_trainer.test(fs_module, episodic_batcher.test_dataloader())
