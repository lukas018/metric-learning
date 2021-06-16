#!/usr/bin/env python3
#!/usr/bin/env python3

import torch
from pytorch_lightning.plugins import  DDPPlugin
from dataclasses import asdict
from metric_learning.algorithms.rfs_lightning import BornAgainLightningModule, DistillArguments
import argparse

import learn2learn as l2l
import pytorch_lightning as pl
from metric_learning.algorithms.rfs import RepresentationForFS, RFSClassifier
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
from metric_learning.utils import HfArgumentParser

parser = HfArgumentParser({PreTrainArguments: "pt", FewshotArguments: "fs", DistillArguments: "dl"})
parser = pl.Trainer.add_argparse_args(parser)

# Define the dataclasses
pt_args: PreTrainArguments
fs_args: FewshotArguments
distill_args: DistillArguments
args: argparse.Namespace

# Parse the dataclasses
pt_args, fs_args, distill_args, args = parser.parse_args_into_dataclasses()

# Initialize the data
datasets, metadatasets, task_datasets = initialize_mini_imagenet(fs_args)
train_ds, base_val_ds = split_mini_imagenet(datasets[0], 0.95)

feat_extractor = l2l.vision.models.ResNet12(10).features
rfs = RepresentationForFS(feat_extractor, RFSClassifier.LogisticRegression)

# Initiailize the pretrainer and train
pt_module = LightningPretrainModule(
    rfs, pt_args, dimensions=640, num_classes=64
)

pretrain_data_module = pl.LightningDataModule.from_datasets(
    train_ds,
    base_val_ds,
    batch_size=pt_args.batch_size,
    num_workers=pt_args.num_workers,
)

pt_trainer = pl.Trainer(
    **{
        **vars(args),
        'gpus': pt_args.gpus,
        'accumulate_grad_batches': pt_args.accumulate_grad_batches,
        'max_epochs': pt_args.n_epochs,
        'default_root_dir': pt_args.checkpoint,
        'plugins': DDPPlugin(find_unused_parameters=False)
    }
)

pt_trainer.fit(pt_module, pretrain_data_module)

# Load the checkpoint
# pt_module = LightningPretrainModule.load_from_checkpoint(
#     "abcd/lightning_logs/version_0/checkpoints/epoch=0-step=142.ckpt",
# )

# Do distillation
teacher = pt_module.metric_model
for i in range(distill_args.n_generations):
    # Initialize a new student model
    student = l2l.vision.models.ResNet12(10).features
    student = RepresentationForFS(student, RFSClassifier.LogisticRegression)
    student.init_pretraining(640, 64)
    distill_data_module = pl.LightningDataModule.from_datasets(
        train_ds,
        base_val_ds,
        batch_size=distill_args.batch_size,
        num_workers=distill_args.num_workers,
    )

    # Run a training session with the distillation module
    distill_module = BornAgainLightningModule(
        training_arguments=pt_args,
        teacher=teacher,
        student=student,
        gamma=distill_args.gamma,
        alpha=distill_args.alpha,
        beta=distill_args.beta
    )

    distill_trainer = pl.Trainer(
        **{
            **vars(args),
            'gpus': distill_args.gpus,
            'default_root_dir': f"{distill_args.checkpoint}-gen-{i}",
            'accumulate_grad_batches': pt_args.accumulate_grad_batches,
            'max_epochs': pt_args.n_epochs,
            # 'plugins': DDPPlugin(find_unused_parameters=False)
        }
    )

    distill_trainer.fit(distill_module, distill_data_module)

    # Let the student become the teacher now
    teacher = distill_module.student

torch.save(teacher, f"rfs-distilled-gen{distill_args.n_generations - 1}")
