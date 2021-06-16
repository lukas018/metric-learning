#!/usr/bin/env python3
import sys
from pathlib import Path
import yaml
import argparse
from itertools import accumulate

import learn2learn as l2l
import pytorch_lightning as pl
from learn2learn.utils.lightning import NoLeaveProgressBar, TrackTestAccuracyCallback
from metric_learning.algorithms import MetaBaseline
from metric_learning.algorithms.utils_lightning import (
    LightningFewshotModule,
    LightningPretrainModule,
)
from metric_learning.arguments import FewshotArguments, PreTrainArguments
from metric_learning.data import (
    EpisodicBatcher,
    initialize_mini_imagenet,
    split_mini_imagenet,
)
from metric_learning.utils import HfArgumentParser, parse_arg_file

parser = HfArgumentParser({PreTrainArguments: "pt", FewshotArguments: "fs"})
parser = pl.Trainer.add_argparse_args(parser)
pt_args, fs_args, args = parser.parse_args_into_dataclasses()
datasets, metadatasets, task_datasets = initialize_mini_imagenet(fs_args)

# Take out the feature extractor from ResNet12
feat_extractor = l2l.vision.models.ResNet12(10).features

# Wrap the feature extractor with the metric learner
metric_learner = MetaBaseline(feat_extractor)


pt_trainer_args = dict(
    gpus=pt_args.gpus,
    weights_save_path=pt_args.checkpoint,
    accumulate_grad_batches=4,
    default_root_dir="pre-trainer",
)


pre_trainer = pl.Trainer(
    **pt_trainer_args,
)

# Split the pretraing dataset to check validation of pretraining
train_ds, base_val_ds = split_mini_imagenet(datasets[0], 0.95)
pt_module = LightningPretrainModule(
    metric_learner, pt_args, dimensions=640, num_classes=len(set(train_ds.y))
)
pretrain_data_module = pl.LightningDataModule.from_datasets(
    train_ds,
    base_val_ds,
    batch_size=pt_args.batch_size,
    num_workers=pt_args.num_workers,
)

# pre_trainer.fit(model=pt_module, datamodule=pretrain_data_module)
# pre_trainer.validate()

meta_trainer = pl.Trainer(
    gpus=fs_args.gpus,
    weights_save_path=fs_args.checkpoint,
    accumulate_grad_batches=16,
)

fs_module = LightningFewshotModule(metric_learner, fs_args)
episodic_batcher = EpisodicBatcher(*task_datasets, epoch_length=fs_args.epoch_length)
meta_trainer.fit(model=fs_module, datamodule=episodic_batcher)
meta_trainer.test(ckpt_path="best")
meta_trainer.save_checkpoint("fewshot-trainer/final.ckpt")
