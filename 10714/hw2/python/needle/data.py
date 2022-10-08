import gzip
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if (flip_img):
            return np.flip(img, axis=1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, 
            high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        min_x = shift_x + self.padding
        min_y = shift_y + self.padding

        # print(">>>", self.padding, img.shape, shift_x, shift_y)
        # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        return np.pad(img, self.padding)[
            min_x : min_x + img.shape[0],
            min_y : min_y + img.shape[1],
            self.padding : self.padding + img.shape[2],
        ]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
        # print(">>>", self.ordering)

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.n = 0
        ### END YOUR SOLUTION
        return self

    # https://www.programiz.com/python-programming/iterator
    """ Test-code
    for i, batch in enumerate(train_dataloader):
        batch_x  = batch[0].numpy()
        truth_x = train_dataset[i * batch_size:(i + 1) * 
            batch_size][0].reshape((batch_size,10,10))
        np.testing.assert_allclose(truth_x, batch_x)
    """
    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.n >= len(self.dataset): raise StopIteration
        curr = self.n
        self.n += 1

        order = self.ordering[curr]
        batch = [Tensor(self.dataset[i]) for i in order]
        # batch = [self.dataset[i] for i in order]
        print(">>>", order, len(batch))
        return batch
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        self.image_filename = image_filename
        self.label_filename = label_filename

        with gzip.open(self.image_filename) as f:
            pixels = np.frombuffer(f.read(), 'B', offset=16) # skip first 16-bytes
            self.images = pixels.reshape(-1, 28*28).astype('float32') / 255

        with gzip.open(self.label_filename) as f:
            self.labels = np.frombuffer(f.read(), 'B', offset=8) # skip first 8-bytes

        self.cached_len = len(self.images)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index]
        if self.transforms:
            img = img.reshape((28, 28, 1))
            img = self.apply_transforms(img)
        return (img, self.labels[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.cached_len
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
