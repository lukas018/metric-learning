#!/usr/bin/env bash
python rfs_evaluate.py \
    --model-checkpoint "rfs-distilled-gen1" --fs_shots 1 --fs_ways 5 --fs_gpus 0
