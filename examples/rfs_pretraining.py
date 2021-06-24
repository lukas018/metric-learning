#!/usr/bin/env python3
#!/usr/bin/env python3

from pathlib import Path
from pytorch_lightning.accelerators import accelerator
import torch
from pytorch_lightning.plugins import  DDPPlugin
from metric_learning.algorithms.rfs_lightning import BornAgainLightningModule, DistillArguments

import pytorch_lightning as pl
from metric_learning.algorithms.rfs import RepresentationForFS, RFSClassifier
from metric_learning.algorithms.utils_lightning import (
    LightningPretrainModule,
)
from metric_learning.arguments import FewshotArguments, PreTrainArguments
from metric_learning.data import (
    initialize_mini_imagenet,
    split_mini_imagenet,
)
from metric_learning.models.resnet import resnet12

# Define the dataclasses
pt_args = PreTrainArguments(
    batch_size=64,
    accumulate_grad_batches=1,
    lr=0.05,
    weight_decay=0.0005,
    n_epochs=100,
    milestones=[40, 80],
    checkpoint="rfs-models",
)

distill_args = DistillArguments(
    batch_size=64,
    accumulate_grad_batches=1,
    lr=0.05,
    weight_decay=0.0005,
    alpha=0.5,
    beta=0.5,
    n_generations=2,
    n_epochs=100,
    milestones=[40, 80],
    checkpoint="rfs-models-distill",
)

# Initialize the data
datasets, metadatasets, task_datasets = initialize_mini_imagenet(FewshotArguments())
train_ds, base_val_ds = split_mini_imagenet(datasets[0], 0.98)

feat_extractor = resnet12(avg_pool=True)
rfs = RepresentationForFS(feat_extractor, RFSClassifier.LogisticRegression)
# Initiailize the pretrainer and train
pt_module = LightningPretrainModule(
    rfs, pt_args, dimensions=640, num_classes=64
)

from pytorch_lightning.callbacks import LearningRateMonitor
lr_monitor = LearningRateMonitor(logging_interval='step')
pretrain_data_module = pl.LightningDataModule.from_datasets(
    train_ds,
    base_val_ds,
    batch_size=pt_args.batch_size,
    num_workers=pt_args.num_workers,
)

pt_trainer = pl.Trainer(
    gpus=1,
    accumulate_grad_batches = pt_args.accumulate_grad_batches,
    default_root_dir="rfs-pretrain",
    max_epochs=pt_args.n_epochs,
    plugins= DDPPlugin(find_unused_parameters=False),
    callbacks=[lr_monitor],
    accelerator="ddp"
)

pt_trainer.fit(pt_module, pretrain_data_module)
teacher = pt_module.metric_model
Path(str(distill_args.checkpoint)).mkdir(exist_ok=True)
torch.save(teacher, Path(str(distill_args.checkpoint), f"rfs-distilled-model-org" ))

# Do distillation
for i in range(distill_args.n_generations):
    # Initialize a new student model
    student = resnet12(avg_pool=True)
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
        beta=distill_args.beta,
    )

    distill_trainer = pl.Trainer(
        gpus=1,
        default_root_dir=f"{distill_args.checkpoint}-gen-{i}",
        accumulate_grad_batches=pt_args.accumulate_grad_batches,
        max_epochs=pt_args.n_epochs,
        callbacks=[lr_monitor],
        accelerator="ddp"
    )
    

    distill_trainer.fit(distill_module, distill_data_module)

    # Let the student become the teacher now
    teacher = distill_module.student
    torch.save(teacher, str(Path(str(distill_args.checkpoint), f"rfs-distilled-model{i}" )))
