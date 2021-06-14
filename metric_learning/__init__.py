__version__ = '0.1.0'

import torch
import learn2learn as l2l
from torch.utils.data import Dataset
from learn2learn.data.meta_dataset import MetaDataset
from learn2learn.utils.lightning import EpisodicBatcher


def test_fn(a: int, b: int) -> torch.Tensor:
    """TODO describe function

    :param a: This is one of the things
    :param b: This is another of the things
    :returns:

    """

    return torch.tensor([a, b])


if __name__ == '__main__':
    a: int = 0
    b: int = 2
    b += 1
    print(f"Hello world={a + b}")
    print("Hello again")
