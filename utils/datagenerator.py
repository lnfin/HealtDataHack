import os
import random
import numpy as np

import albumentations as A
from .dataset import CancerDataset
from torch.utils.data import DataLoader


def get_paths(directories):
    """
    :param directories: list of directories or one directory, where data is
    :return: [[image_path, mask_path], ...]
    """
    if isinstance(directories, str):
        directories = [directories]
    data = []
    for directory in directories:
        paths = os.listdir(directory)
        # getting only image names without extension
        image_names = [path.split('/')[-1].split('.')[0] for path in paths if 'mask' not in path]
        for name in image_names:
            image_path = os.path.join(directory, name + '.jpg')
            mask_path = os.path.join(directory, name + '_mask.jpg')

            # checking for existing
            if not os.path.exists(image_path):
                print(f'{image_path} does not exist')
                continue
            if not os.path.exists(mask_path):
                print(f'{mask_path} does not exist')
                continue

            data.append([os.path.join(directory, name + '.jpg'),
                         os.path.join(directory, name + '_mask.jpg')])
    return data


def data_generator(cfg):
    # getting train and val paths
    train_paths = get_paths(cfg.data_folders_train)
    val_paths = get_paths(cfg.data_folders_val)

    # type correction
    train_paths = np.asarray(train_paths)
    val_paths = np.asarray(val_paths)

    # shuffling
    random.shuffle(train_paths)
    random.shuffle(val_paths)
    return train_paths, val_paths


def get_transforms(cfg):
    # getting transforms from albumentations
    pre_transforms = [getattr(A, item["name"])(**item["params"]) for item in cfg.pre_transforms]
    augmentations = [getattr(A, item["name"])(**item["params"]) for item in cfg.augmentations]
    post_transforms = [getattr(A, item["name"])(**item["params"]) for item in cfg.post_transforms]

    # concatenate transforms
    train = A.Compose(pre_transforms + augmentations + post_transforms)
    test = A.Compose(pre_transforms + post_transforms)
    return train, test


def get_loaders(cfg):
    # getting transforms
    train_transforms, test_transforms = get_transforms(cfg)

    # getting train and val paths
    train_paths, val_paths = data_generator(cfg)

    # creating datasets
    train_ds = CancerDataset(train_paths, transform=train_transforms)
    val_ds = CancerDataset(val_paths, transform=train_transforms)

    # creating data loaders
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, drop_last=True)
    return train_dl, val_dl
