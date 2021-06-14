#!/usr/bin/env python3

import argparse
import pytorch_lightning as pl
import learn2learn as l2l
from learn2learn.utils.lightning import EpisodicBatcher
from metric_learning.algorithms import MetaBaseline
from metric_learning.algorithms.lightning import LightningFewshotModule, LightningPretrainModule, PreTrainArguments, FewshotArguments
from metric_learning.utils import parse_arg_file
from metric_learning.data import initialize_mini_imagenet, split_mini_imagenet

# Take out the feature extractor from ResNet12
feat_extractor = l2l.vision.models.ResNet12(10).features

# Wrap the feature extractor with the metric learner
metric_learner = MetaBaseline(feat_extractor)

parser = argparse.ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument('--config-file', type=str, required=True)
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--fs_train', action='store_true')

# Initialize the pretraining
args = parser.parse_args()
pt_args, fs_args = parse_arg_file(args.config_file)
datasets, metadatasets, task_datasets = initialize_mini_imagenet(fs_args)

if args.pretrain:
    pre_trainer = pl.Trainer(args, weights_save_path=pt_args.checkpoint)
    pt_module = LightningPretrainModule(metric_learner, pt_args)

    # Split the pretraing dataset to check validation of pretraining
    train_ds, base_val_ds = split_mini_imagenet(datasets[0], 0.95)
    pretrain_data_module = pl.LightningDataModule.from_datasets(train_ds, base_val_ds)
    pre_trainer.fit(model=pt_module, datamodule=pretrain_data_module)
    pre_trainer.validate()


if args.metatrain:
    meta_trainer = pl.Trainer(
        args,
        callbacks=[
            l2l.utils.lightning.TrackTestAccuracyCallback(),
            l2l.utils.lightning.NoLeaveProgressBar(),
        ]
    )

    fs_module = LightningFewshotModule(metric_learner, pt_args)
    episodic_batcher = EpisodicBatcher(*task_datasets)
    meta_trainer.fit(model=fs_module, datamodule=episodic_batcher)
    meta_trainer.test(ckpt_path="best")
    meta_trainer.save_checkpoint("final.ckpt")
