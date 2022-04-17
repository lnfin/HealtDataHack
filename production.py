import torch
from torch import nn
from torch import squeeze, unsqueeze

from utils.cfgtools import get_model
from utils.datagenerator import get_transforms
from modeling.cutter import Cutter

import albumentations as A


class ProductionModel(nn.Module):
    def __init__(self, cfg, cut_size=1024, weights=None, mini_batch_size=4):
        """
        Production model is a model to easy implement to production code
        :param cfg: config of model
        :param cut_size: size of window for cutter
        :param weights: could be 'best_loss', 'best_score' or path to model weights
        """
        super(ProductionModel, self).__init__()
        self.mini_batch_size = mini_batch_size
        self.device = torch.device(cfg.device)
        _, self.transform = get_transforms(cfg)

        # TODO: make data-independent transforms (get it from cfg) and cutter only if needed
        self.cutter = Cutter(
            (cut_size, cut_size),
            (cut_size, cut_size)
        )

        t = A.Resize(512, 512)
        pre_correction = lambda img: img.permute(1, 2, 0).cpu().numpy()
        post_correction = lambda img: unsqueeze(torch.from_numpy(img).permute(2, 0, 1), dim=0)
        self.transform = lambda img: post_correction(t(image=pre_correction(img))['image'])

        inv_t = A.Resize(cut_size, cut_size)
        inv_pre_correction = lambda img: squeeze(img, dim=0).cpu().numpy().astype('float32')
        inv_post_correction = lambda img: unsqueeze(unsqueeze(torch.from_numpy(img), dim=0).expand(3, -1, -1), dim=0)
        self.inv_transform = lambda img: inv_post_correction(inv_t(image=inv_pre_correction(img))['image'])

        if weights == 'best_loss' or weights is None:
            weights = cfg.best_weights[0]
        elif weights == 'best_score':
            weights = cfg.best_weights[1]

        self.model = get_model(cfg)(cfg=cfg).to(self.device)
        self.model.load_state_dict(torch.load(weights))
        self.model.eval()

    def forward(self, image):
        image = squeeze(image, dim=0)
        image = image / 255
        windows = self.cutter.split(image.permute(1, 2, 0))
        windows = torch.cat([self.transform(image) for image in windows])
        mask_list = []
        for i in range(0, windows.shape[0], self.mini_batch_size):
            X_mini_batch = windows[i:i + self.mini_batch_size]
            X_mini_batch = X_mini_batch.to(self.device)
            with torch.no_grad():
                masks = self.model(X_mini_batch)
            for mask in masks:
                mask_list.append(self.inv_transform(mask))

        orig_mask = self.cutter.merge(torch.cat(mask_list))
        orig_mask = orig_mask[0]
        orig_mask = orig_mask.unsqueeze(0)
        return orig_mask
