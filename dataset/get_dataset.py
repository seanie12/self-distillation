import io
import os
import pickle

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

NUM_CLASSES = {
    "aircraft": 100,
    "cub": 200,
    "cifar10": 100,
    "stanford_dogs": 120,
    "dtd": 47,
    "vgg_flower_102": 102,
    "chest_xray": 2,
    "cifar10": 10,
}


def get_dataset(data: str, train: bool, transform=None, target_transform=None):
    if data == "cifar10":
        return CIFAR10("data/cifar10", train, transform, target_transform, download=True)
    else:
        path = f"data/{data}/{'train' if train else 'test'}.lmdb"
        if not os.path.exists(path):
            raise NotImplementedError
        return ImageFolderLMDB(path, transform, target_transform)


class ImageFolderLMDB(Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        env = lmdb.open(
            db_path, subdir=os.path.isdir(db_path), readonly=True, lock=False, readahead=False, meminit=False
        )
        with env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))
        env.close()

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        if not hasattr(self, "env"):
            self.env = lmdb.open(
                self.db_path, subdir=os.path.isdir(self.db_path),
                readonly=True, lock=False, readahead=False, meminit=False
            )
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        imgbuf, target = pickle.loads(byteflow)

        # load image
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"
