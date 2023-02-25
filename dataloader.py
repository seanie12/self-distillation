
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoFeatureExtractor
from dataset.get_dataset import NUM_CLASSES, get_dataset
import numpy as np


def collate_fn(data):
    x, y = zip(*data)
    x = torch.cat([torch.tensor(np.array(_x["pixel_values"])) for _x in x])
    y = torch.tensor(y)
    return x, y


def get_dataloader(name, batch_size, num_workers=1,
                   train_val_split=0.75, seed=42, model_name="facebook/vit-mae-base"):
    transform = AutoFeatureExtractor.from_pretrained(model_name)
    train_ds = get_dataset(name, train=True, transform=transform)
    lengths = [int(train_val_split * len(train_ds)),
               len(train_ds) - int(train_val_split * len(train_ds))]
    if train_val_split < 1.0:
        train_ds, val_ds = random_split(
            train_ds, lengths, torch.Generator().manual_seed(seed))
    else:
        val_ds = None
    test_ds = get_dataset(name, train=False, transform=transform)

    kwargs = {"batch_size": batch_size, "num_workers": num_workers,
              "collate_fn": collate_fn, "pin_memory": True}

    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=True, **kwargs) if val_ds is not None else None
    test_loader = DataLoader(test_ds, **kwargs)

    return train_loader, val_loader, test_loader, NUM_CLASSES[name]

