#!/usr/bin/env python3
import torch
from metric_learning.arguments import FewshotArguments
from metric_learning.utils import HfArgumentParser
import pytorch_lightning as pl
import argparse
from metric_learning.algorithms.utils_lightning import (
    LightningFewshotModule,
)


import learn2learn as l2l
from metric_learning.data import EpisodicBatcher, initialize_mini_imagenet
from metric_learning.algorithms.rfs import RepresentationForFS, RFSClassifier

parser = HfArgumentParser({FewshotArguments:"fs"})
parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument("--model-checkpoint")

fs_args: FewshotArguments
args: argparse.Namespace
fs_args, args = parser.parse_args_into_dataclasses()
datasets, metadatasets, task_datasets = initialize_mini_imagenet(FewshotArguments())
episodic_batcher = EpisodicBatcher(*task_datasets, epoch_length=fs_args.epoch_length)
# feat_extractor = l2l.vision.models.ResNet12(10).features


# rfs = RepresentationForFS(feat_extractor, RFSClassifier.LogisticRegression)
rfs = torch.load(args.model_checkpoint)
fs_module = LightningFewshotModule(rfs, fs_arguments=fs_args)
fs_trainer = pl.Trainer(
    gpus=fs_args.gpus,
    default_root_dir=fs_args.checkpoint
)
# We don't need to train here
for _ in range(10):
    results = fs_trainer.test(fs_module, episodic_batcher.test_dataloader(), verbose=False)
    print(results)
