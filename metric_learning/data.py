#!/usr/bin/env python3

import functools
import copy
import random
from collections import defaultdict
from itertools import chain, starmap
from operator import attrgetter
from typing import Callable, Iterable


import learn2learn as l2l
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from metric_learning.algorithms.lightning.metric_module import FewshotArguments
from torchvision.transforms import Compose, Normalize, ToPILImage, ToTensor


def initialize_mini_imagenet(
    fs_args: FewshotArguments,
    num_tasks=-1,
    data_augmentaion="lee2019"
):
    meta_datasets, tasks = l2l.vision.benchmarks.mini_imagenet_tasksets(
        fs_args.ways,
        fs_args.shots + fs_args.queries,
        fs_args.ways,
        fs_args.shots + fs_args.queries,
        data_augmentation=data_augmentaion
    )

    datasets = list(map(attrgetter("dataset"), meta_datasets))

    def _create_taskdataset(ds, transform):
        return l2l.data.TaskDataset(ds, transform, num_tasks=num_tasks)

    tasks = list(starmap(_create_taskdataset, zip(meta_datasets, tasks)))

    if data_augmentaion is not None:
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

    return datasets, meta_datasets, tasks

def _groupby(iterable: Iterable, key: Callable) -> dict:
    dd = defaultdict(list)
    for elem in iterable:
        dd[key(elem)].append(elem)
    return dd

def _random_split(indices: list[int], split: float) -> tuple[list[int], list[int]]:
    random.shuffle(indices)
    cutoff = len(indices) * split
    cutoff = int(cutoff)
    return indices[:cutoff], indices[cutoff:]

def split_mini_imagenet(ds, split: float = 0.9) -> tuple[Dataset, Dataset]:

    """TODO describe function

    :param ds:
    :param split:
    :returns:

    """
    groups = _groupby(range(len(ds)), key=lambda i: ds.y[i])

    indices = groups.values()
    _split = functools.partial(_random_split, split=split)
    indices1, indices2 = (*zip(*map(_split, indices)),)

    def _split_dataset(indices: list[int]) -> Dataset:
        idx = np.array(indices)
        ds_copy = copy.copy(ds)
        ds_copy.x = ds_copy.x[idx]
        ds_copy.y = ds_copy.y[idx]
        return ds_copy

    ds1 = _split_dataset(list(chain(*indices1)))
    ds2 = _split_dataset(list(chain(*indices2)))
    return ds1, ds2

def split_vbs3_dataset(ds, split: float = 0.95):
    from operator import itemgetter
    labels = list(map(itemgetter(1), ds.samples))
    groups = _groupby(range(len(ds)), key=lambda i: labels[i])
    indices = groups.values()
    _split = functools.partial(_random_split, split=split)
    indices1, indices2 = (*zip(*map(_split, indices)),)

    def _split_dataset(indices: list[int]) -> Dataset:
        idx = np.array(indices)
        ds_copy = copy.copy(ds)
        ds_copy.samples = ds.samples[idx]
        return ds_copy

    ds1 = _split_dataset(list(chain(*indices1)))
    ds2 = _split_dataset(list(chain(*indices2)))
    return ds1, ds2

class EpisodicBatcher(pl.LightningDataModule):

    """
    nc
    """

    def __init__(
        self,
        train_tasks,
        validation_tasks=None,
        test_tasks=None,
        epoch_length=1,
    ):
        super(EpisodicBatcher, self).__init__()
        self.train_tasks = train_tasks
        if validation_tasks is None:
            validation_tasks = train_tasks
        self.validation_tasks = validation_tasks
        if test_tasks is None:
            test_tasks = validation_tasks
        self.test_tasks = test_tasks
        self.epoch_length = epoch_length

    @staticmethod
    def epochify(taskset, epoch_length):
        class Epochifier(Dataset):
            def __init__(self, tasks, length):
                self.tasks = tasks
                self.length = length

            def __getitem__(self, *args, **kwargs):
                return self.tasks.sample()

            def __len__(self):
                return self.length

        return DataLoader(
            Epochifier(taskset, epoch_length), num_workers=8, batch_size=None
        )

    def train_dataloader(self):
        return EpisodicBatcher.epochify(
            self.train_tasks,
            self.epoch_length,
        )

    def val_dataloader(self):
        return EpisodicBatcher.epochify(
            self.validation_tasks,
            self.epoch_length,
        )

    def test_dataloader(self):
        length = self.epoch_length
        return EpisodicBatcher.epochify(
            self.test_tasks,
            length,
        )
