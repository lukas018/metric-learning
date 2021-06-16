#!/usr/bin/env bash
python rfs_pretraining.py \
    --pt_gpus 0 --pt_n_epochs 100 --pt_lr 0.05 --pt_weight_decay 0 --pt_batch_size 64 --pt_checkpoint rfs-petraining \
    --dl_gpus 0 --dl_n_epochs 100 --dl_lr 0.05 --dl_alpha 0.5 --dl_beta 0.5 --dl_batch_size 64 --dl_n_generations 2 --dl_checkpoint rfs-distill\
    --accelerator ddp
    # --fs_gpus 0 --fs_shots 1 --fs_ways 10
