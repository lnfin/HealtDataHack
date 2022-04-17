import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CancerDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        self._len = len(self.paths)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        paths = self.paths[index]
        if isinstance(paths, str):
            image_path, mask_path = paths, None
        elif isinstance(paths, tuple):
            image_path, mask_path = paths
        elif isinstance(paths, list):
            image_path, mask_path = paths
        elif isinstance(paths, np.ndarray):
            image_path, mask_path = paths
        else:
            raise TypeError(f"Unsupported type {type(paths)} for loading images")

        # read in rgb
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mask_path:
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = np.expand_dims(mask, axis=-1)
        else:
            mask = np.zeros_like(image)

        # using transform if needed
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # to [0..1]
        image = image / 255
        mask = mask > 128  # thresholding

        # data type and dimension correction
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(np.array(image, dtype=np.float))
        image = image.type(torch.FloatTensor)
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        return image, mask
