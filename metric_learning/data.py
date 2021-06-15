#!/usr/bin/env python3

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


def initialize_mini_imagenet(
    fs_args: FewshotArguments,
    num_tasks=-1,
):
    print(type(fs_args.ways))
    meta_datasets, tasks = l2l.vision.benchmarks.mini_imagenet_tasksets(
        fs_args.ways,
        fs_args.shots + fs_args.queries,
        fs_args.ways,
        fs_args.shots + fs_args.queries,
        data_augmentation="lee2019"
    )

    datasets = list(map(attrgetter("dataset"), meta_datasets))

    def _create_taskdataset(ds, transform):
        return l2l.data.TaskDataset(ds, transform, num_tasks=num_tasks)

    tasks = list(starmap(_create_taskdataset, zip(meta_datasets, tasks)))
    return datasets, meta_datasets, tasks


def split_mini_imagenet(ds, split: float = 0.9) -> tuple[Dataset, Dataset]:

    """TODO describe function

    :param ds:
    :param split:
    :returns:

    """

    def groupby(iterable: Iterable, key: Callable) -> dict:
        dd = defaultdict(list)
        for elem in iterable:
            dd[key(elem)].append(elem)
        return dd

    groups = groupby(range(len(ds)), key=lambda i: ds.y[i])

    def random_split(indices: list[int]) -> tuple[list[int], list[int]]:
        random.shuffle(indices)
        cutoff = len(indices) * split
        cutoff = int(cutoff)
        return indices[:cutoff], indices[cutoff:]

    indices = groups.values()
    indices1, indices2 = (*zip(*map(random_split, indices)),)

    def _split_dataset(indices: list[int]) -> Dataset:
        idx = np.array(indices)
        ds_copy = copy.copy(ds)
        ds_copy.x = ds_copy.x[idx]
        ds_copy.y = ds_copy.y[idx]
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

        return DataLoader(Epochifier(taskset, epoch_length), num_workers=8, batch_size=None)

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
