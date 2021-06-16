#!
# /usr/bin/env python3
import os
import random
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    ColorJitter,
    CenterCrop,
)


def _build_vbs3_dataset(root: str, banned_classes: set, size=84):
    """Load a defaul vbs3 dataset

    :param root: Root of dataset folder
    :param banned_classes: A set of class names to ignore
    :returns: A dataset of VBS3 images

    """
    data_transforms = Compose(
        [
            RandomCrop(size, padding=8),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    )

    class MetaImageFolder(ImageFolder):

        # Override
        def _find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
            classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
            classes = list(filter(lambda entry: entry not in banned_classes, classes))
            if not classes:
                raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            return classes, class_to_idx

    ds = MetaImageFolder(
        root, transform=data_transforms,
    )
    return ds


def build_baseline_dataset(root, size=84):
    """TODO describe function

    :param root:
    :returns:

    """
    data_transforms = Compose(
        [
            RandomCrop(size, padding=8),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            RandomHorizontalFlip(),
            ToTensor(),
            # normalize,
        ]
    )
    ds = ImageFolder(root, transform=data_transforms)
    return ds


def build_test_dataset(root, size=84):
    """TODO describe function

    :param root:
    :returns:

    """
    data_transforms = Compose(
        [
            CenterCrop(size),
            ToTensor(),
        ]
    )
    ds = ImageFolder(root, transform=data_transforms)
    return ds


def generate_class_split(classes, banned_classes, n_train_classes=370):
    available_classes = {cls for cls in classes if not cls in banned_classes}
    training_classes = set(random.sample(list(available_classes), n_train_classes))
    validation_classes = {av for av in available_classes if av not in training_classes}
    return training_classes, validation_classes


def build_vbs3_dataset(root, classes, n_train_classes, seed=42, size=84):
    random.seed(seed)
    banned = {"Lada"}
    train_split, val_split = generate_class_split(classes, banned, n_train_classes)
    train_ds = _build_vbs3_dataset(root, {*val_split, *banned}, size=size)
    val_ds = _build_vbs3_dataset(root, {*train_split, *banned}, size=size)
    return train_ds, val_ds

if __name__ == '__main__':
    from pathlib import Path
    classes = [p.name for p in Path("/mnt/hdd1/Data/vbs3-dataset/").glob("*/")]
    train_ds, val_ds = build_vbs3_dataset("/mnt/hdd1/Data/vbs3-dataset/", classes, 380)
    breakpoint()
    baseline = build_baseline_dataset("/home/luklun/data/baseline")
    test = build_test_dataset("/home/luklun/data/baseline")
