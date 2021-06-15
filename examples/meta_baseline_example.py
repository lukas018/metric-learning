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
from metric_learning.algorithms.lightning import (
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
from torchvision.transforms import Compose, Normalize, ToPILImage, ToTensor

parser = HfArgumentParser({PreTrainArguments: "pt", FewshotArguments: "fs"})
parser = pl.Trainer.add_argparse_args(parser)
# parser.add_argument("--config-file", type=str, required=True)
# parser.add_argument("--pretrain", action="store_true")
# parser.add_argument("--fs_train", action="store_true")
# pt_args, fs_args = parse_arg_file(args.config_file)

breakpoint()
# if any( sys.argv[1].endswith(_ending) for _ending in ('json', 'yaml', 'yml',)):
#     file_content = Path(sys.argv[1]).read_text()
#     if sys.argv[1].endswith("json"):
#         dict_to_parse = json.loads(file_content)
#     else:
#         dict_to_parse = yaml.load(file_content, Loader=yaml.BaseLoader)

#     breakpoint()
#     test = parser.parse_dict(dict_to_parse)
# else:
pt_args, fs_args, args = parser.parse_args_into_dataclasses()
datasets, metadatasets, task_datasets = initialize_mini_imagenet(fs_args)

# Take out the feature extractor from ResNet12
feat_extractor = l2l.vision.models.ResNet12(10).features

# Wrap the feature extractor with the metric learner
metric_learner = MetaBaseline(feat_extractor)

# Fix the normalization bug in learn2learn
# This makes all the datasets, including the task and validation dataset works
normalize = Normalize(
    mean=[120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
    std=[70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0],
)
val_transform = Compose(
    [
        ToPILImage(),
        ToTensor(),
        normalize,
    ]
)
val_ds, test_ds = datasets[1:]
val_ds.transform = val_transform
test_ds.transform = val_transform

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

pre_trainer.fit(model=pt_module, datamodule=pretrain_data_module)
pre_trainer.validate()

meta_trainer = pl.Trainer(
    gpus=args.gpus,
    weights_save_path=pt_args.checkpoint,
    accumulate_grad_batches=16,
    default_root_dir="fewshot-trainer",
)

fs_module = LightningFewshotModule(metric_learner, fs_args)
episodic_batcher = EpisodicBatcher(*task_datasets, epoch_length=fs_args.epoch_length)
meta_trainer.fit(model=fs_module, datamodule=episodic_batcher)
meta_trainer.test(ckpt_path="best")
meta_trainer.save_checkpoint("fewshot-trainer/final.ckpt")
