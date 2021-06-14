#!/usr/bin/env python3

import copy
import random
from collections import defaultdict
from itertools import chain, starmap
from operator import attrgetter
from typing import Callable, Iterable

import learn2learn as l2l
import numpy as np
from torch.utils.data import Dataset


def initialize_mini_imagenet(
    fs_args,
    num_tasks=-1,
):
    meta_datasets, tasks = l2l.vision.benchmarks.mini_imagenet_tasksets(
        fs_args.ways, fs_args.shots + fs_args.queries, data_augmentation="lee2019"
    )
    breakpoint()

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
        return ds

    ds1 = _split_dataset(list(chain(*indices1)))
    ds2 = _split_dataset(list(chain(*indices2)))
    return ds1, ds2
