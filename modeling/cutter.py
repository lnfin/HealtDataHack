import torch
import torch.nn.functional as F
import cv2
import numpy as np


def is_enough_presentable(image, threshold=0.5):
    """
    Filter image by histogram value
    :param image: image to check
    :param threshold: threshold of filtering [0..1]
    :return: boolean value is image enough presentable or not
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    histogram, _ = np.histogram(grayscale, bins=100, range=(0.0, 1.0))
    if np.sum(histogram[:80]) / np.sum(histogram) >= threshold:
        return True
    return False


class Cutter:
    def __init__(self, kernel_size=(1024, 1024), stride=(1024, 1024)):
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h, self.stride_w = stride

    def split(self, image):
        """
        Split image to many little patches
        :param image: torch.Tensor of (H, W, C) format
        :return: torch.Tensor of (N, C, H, W) format
        """

        self.orig_size = image.shape[:2]

        # padding of right and bottom sides of image
        pad_size = (self.kernel_w - self.orig_size[1] % self.kernel_w) % self.kernel_w
        padded_image = F.pad(image.permute(2, 0, 1), (0, pad_size), "constant", 0)
        pad_size = (self.kernel_h - self.orig_size[0] % self.kernel_h) % self.kernel_h
        padded_image = F.pad(padded_image.permute(0, 2, 1), (0, pad_size), "constant", 0)

        # dimension correction (to (B, C, H, W) format)
        tensor_image = torch.unsqueeze(padded_image, dim=0).permute(0, 1, 3, 2)

        # splitting
        windows = tensor_image.unfold(2, self.kernel_h, self.stride_h).unfold(3, self.kernel_w, self.stride_w)
        self.unfold_shape = windows.size()

        # dimension correction (to (N, C, H, W) format)
        windows = windows.permute(2, 3, 0, 1, 4, 5).reshape(-1, 3, self.kernel_h, self.kernel_w)
        return windows

    def merge(self, windows):
        """
        Merge many mask patches to mask of original image size
        :param windows: patches to merge of format (N, C, H, W)
        :return: mask of original image size (1, C, H, W)
        """
        window_number_h = self.unfold_shape[2]
        window_number_w = self.unfold_shape[3]
        _, c, window_h, window_w = windows.size()
        x_image = windows.view(1, window_number_h, window_number_w, c, window_h, window_w)
        x_image = x_image.permute(0, 3, 1, 4, 2, 5).contiguous()
        x_image = x_image.view(1, c, window_number_h * window_h, window_number_w * window_w)
        return x_image[0, :, :self.orig_size[0], :self.orig_size[1]]
