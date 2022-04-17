import torch
import numpy as np


class IoUScore(object):
    def __init__(self, eps=1e-7):
        self.eps = eps

    def __call__(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, str(y_pred.shape) + ' != ' + str(y_true.shape)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
            y_pred = torch.from_numpy(y_pred)
            y_true = y_true.unsqueeze(0)
            y_pred = y_pred.unsqueeze(0)
        intersection = torch.sum(y_true * y_pred >= 1, dim=[1, 2])
        union = torch.sum(y_true, dim=[1, 2]) + torch.sum(y_pred, dim=[1, 2]) - intersection
        iou = ((intersection + self.eps) / (union + self.eps)).mean(dim=0)
        return iou


class DiceScore(object):
    def __init__(self, eps=1e-7):
        self.eps = eps

    def __call__(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, str(y_pred.shape) + ' != ' + str(y_true.shape)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
            y_pred = torch.from_numpy(y_pred)
            y_true = y_true.unsqueeze(0)
            y_pred = y_pred.unsqueeze(0)
        intersection = torch.sum(y_true * y_pred >= 1, dim=[1, 2])
        union = torch.sum(y_true >= 1, dim=[1, 2]) + torch.sum(y_pred >= 1, dim=[1, 2])
        dice = ((2 * intersection + self.eps) / (union + self.eps)).mean(dim=0)
        return dice