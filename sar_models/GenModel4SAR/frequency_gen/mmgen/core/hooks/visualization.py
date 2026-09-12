import os.path as osp

import mmcv
import torch
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only
from skimage import io
from torchvision.utils import save_image


@HOOKS.register_module('MMGenVisualizationSARHook')
class VisualizationSARHook(Hook):
    """Save SAR training samples as TIFF plus RGB preview images."""

    def __init__(
        self,
        output_dir,
        res_name_list,
        interval=-1,
        filename_tmpl='iter_{}.tiff',
        nrow=2,
        padding=2,
        pad_value=1,
        n_channel=6,
        n_edge=32,
    ):
        assert mmcv.is_list_of(res_name_list, str)
        self.output_dir = output_dir
        self.res_name_list = res_name_list
        self.interval = interval
        self.filename_tmpl = filename_tmpl
        stem, _ = osp.splitext(filename_tmpl)
        self.filename_tmpl_rgb = f'{stem}.png'
        self.nrow = nrow
        self.padding = padding
        self.pad_value = pad_value
        self.n_channel = n_channel
        self.n_edge = n_edge

    @master_only
    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        results = runner.outputs['results']
        tensors = [
            self._crop(results[key]) for key in self.res_name_list
            if key in results
        ]
        if not tensors:
            return

        if not hasattr(self, '_out_dir'):
            self._out_dir = osp.join(runner.work_dir, self.output_dir)
        mmcv.mkdir_or_exist(self._out_dir)

        img_cat = torch.cat(tensors, dim=3).detach().cpu()
        iteration = runner.iter + 1
        io.imsave(osp.join(self._out_dir, self.filename_tmpl.format(iteration)),
                  img_cat.numpy())

        if img_cat.shape[1] == 3:
            preview = self.zero_one_rgb(img_cat)
        elif img_cat.shape[1] == 6:
            preview = torch.cat(
                [self.zero_one_rgb(img_cat[:, 0:3, ...]),
                 self.zero_one_rgb(img_cat[:, 3:6, ...])],
                dim=0)
        else:
            preview = img_cat[:, :3, ...].clamp(0, 1)

        save_image(
            preview,
            osp.join(self._out_dir, self.filename_tmpl_rgb.format(iteration)),
            nrow=self.nrow,
            padding=self.padding,
            pad_value=self.pad_value)

    def _crop(self, tensor):
        tensor = tensor[:, 0:self.n_channel, ...]
        if self.n_edge == 0:
            return tensor
        return tensor[:, :, self.n_edge:-self.n_edge, self.n_edge:-self.n_edge]

    @staticmethod
    def zero_one_rgb(img_cat):
        img_cat = img_cat[:, [2, 1, 0], ...]
        img_cat = (img_cat * 0.5 + 0.5).clamp_(0, 1)
        img_cat = torch.pow(img_cat, 3)
        for channel in range(3):
            max_value = torch.quantile(
                img_cat[:, channel, ...], 0.99, interpolation='linear')
            img_cat[:, channel, ...] = img_cat[:, channel, ...] / max_value
        return img_cat.clamp_(0, 1)
