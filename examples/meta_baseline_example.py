import torch
import learn2learn as l2l
import pytorch_lightning as pl
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

pt_args = PreTrainArguments(lr=0.1, weight_decay=0.0005, accumulate_grad_batches=2, batch_size=64, milestones=[90], lr_reduce=0.1, n_epochs=100)
datasets, metadatasets, task_datasets = initialize_mini_imagenet(FewshotArguments())

# Take out the feature extractor from ResNet12
feat_extractor = l2l.vision.models.ResNet12(10).features

# Wrap the feature extractor with the metric learner
metric_learner = MetaBaseline(feat_extractor)

pre_trainer = pl.Trainer(
    default_root_dir="meta-baseline-pretraining",
    max_epochs=pt_args.n_epochs,
    gpus=[0],
    accumulate_grad_batches=pt_args.accumulate_grad_batches
)

# Split the pretraing dataset to check validation of pretraining
train_ds, base_val_ds = split_mini_imagenet(datasets[0], 0.98)
pt_module = LightningPretrainModule(
    metric_learner, pt_args, dimensions=640, num_classes=len(set(train_ds.y)),
)
pretrain_data_module = pl.LightningDataModule.from_datasets(
    train_ds,
    base_val_ds,
    batch_size=pt_args.batch_size,
    num_workers=pt_args.num_workers,
)

pre_trainer.fit(model=pt_module, datamodule=pretrain_data_module)
pre_trainer.validate()

# Save the
torch.save(pt_module.metric_model, f"meta-baseline-pretraining/pretrained.pth")

# meta_trainer = pl.Trainer(
#     gpus=fs_args.gpus,
#     weights_save_path=fs_args.checkpoint,
#     accumulate_grad_batches=16,
# )

# fs_module = LightningFewshotModule(metric_learner, fs_args)
# episodic_batcher = EpisodicBatcher(*task_datasets, epoch_length=fs_args.epoch_length)
# meta_trainer.fit(model=fs_module, datamodule=episodic_batcher)
# meta_trainer.test(ckpt_path="best")
# meta_trainer.save_checkpoint("fewshot-trainer/final.ckpt")
