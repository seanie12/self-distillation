import argparse

import kornia.augmentation as K
import torch
import torch.nn as nn


class InfIterator(object):
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)

    def __len__(self):
        return len(self.iterable)


class Augmentation(nn.Module):
    def __init__(self, img_size) -> None:
        super().__init__()
        self.img_size = img_size

        rnd_resizedcrop = K.RandomCrop(size=(img_size, img_size),
                                       p=1.0, same_on_batch=False)
        rnd_hflip = K.RandomHorizontalFlip(
            p=0.5, same_on_batch=False)

        self.transform = nn.Sequential(
            rnd_resizedcrop,
            rnd_hflip,
        )

    def forward(self, x):
        return self.transform(x)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def accuracy(y, logits):
    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
        acc = 1.0 * pred.eq(y).sum() / y.size(0)
        return acc.item()
