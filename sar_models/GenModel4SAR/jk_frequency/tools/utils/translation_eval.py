import argparse
import os
import os.path as osp
import shutil

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from skimage import io
from torchvision.utils import save_image

from mmgen.apis import set_random_seed
from mmgen.datasets import build_dataloader, build_dataset
from mmgen.models import build_model
from mmgen.models.translation_models import BaseTranslationModel
from mmgen.utils import get_root_logger

EPS = 2.2204e-16
MAX_X = torch.tensor([70, 70, 85, 50, 60, 55])
MAX_KA = torch.tensor([25, 35, 60, 30, 30, 40])
IMAGE_SUFFIXES = ('.jpg', '.png', '.jpeg', '.JPEG', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate SAR translation images')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--target-domain', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--samples-path', type=str, default=None)
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--eval', nargs='*', type=str, default=['none'])
    parser.add_argument('--delete_pre_folder', action='store_true')
    parser.add_argument('--generate_whole', action='store_true')
    parser.add_argument(
        '--field',
        type=str,
        default='all',
        choices=['all', 'real_x', 'real_ka', 'fake'])
    parser.add_argument('--n_edge', type=int, default=32)
    parser.add_argument(
        '--channel_list', nargs='+', type=int, default=[1, 2, 0])
    return parser.parse_args()


@torch.no_grad()
def zero_one_rgb(img_cat, max_norm):
    img_cat = (img_cat * 0.5 + 0.5).clamp_(0, 1)
    for channel in range(img_cat.size(1)):
        img_cat[:, channel, ...] *= max_norm[channel]
    img_cat = torch.pow(img_cat, 3)

    min_rgb = torch.tensor(
        [33.7469, 46.6785, 28.6685, 21.7749, 25.0943, 18.1471])
    max_rgb = torch.tensor(
        [86.8550, 93.6178, 89.4642, 80.4738, 76.9785, 79.9200])
    img_cat = 10 * torch.log(img_cat + EPS)
    for channel in range(img_cat.size(1)):
        torch.clamp_(
            img_cat[:, channel, ...], min_rgb[channel], max_rgb[channel])
        img_cat[:, channel, ...] = (
            (img_cat[:, channel, ...] - min_rgb[channel]) /
            (max_rgb[channel] - min_rgb[channel]))
    return img_cat


def count_test_images(cfg):
    testdir = cfg.data.test.get('testdir', 'testA')
    datafolder = osp.join(cfg.data.test.dataroot, testdir)
    return len(list(mmcv.scandir(datafolder, suffix=IMAGE_SUFFIXES)))


def crop_edges(tensor, n_edge):
    if n_edge == 0:
        return tensor
    return tensor[..., n_edge:-n_edge, n_edge:-n_edge]


def output_name(field, checkpoint_name, suffix):
    if field == 'fake':
        return f'whole_img_fake_{checkpoint_name}.{suffix}'
    if field == 'real_x':
        return f'whole_img_realX.{suffix}'
    return f'whole_img_realKA.{suffix}'


def fields_to_export(field):
    if field == 'all':
        return ['real_ka', 'real_x', 'fake']
    return [field]


def field_batch(field, data_batch, output_dict, source_domain, target_domain):
    if field == 'fake':
        return output_dict['target']
    if field == 'real_ka':
        return data_batch[f'img_{target_domain}']
    return data_batch[f'img_{source_domain}']


def field_norm(field):
    return MAX_X if field == 'real_x' else MAX_KA


@torch.no_grad()
def generate_whole_image(
    model,
    data_loader,
    samples_path,
    checkpoint_name,
    source_domain,
    target_domain,
    batch_size,
    num_samples,
    channel_list,
    n_edge,
    field,
):
    if samples_path:
        mmcv.mkdir_or_exist(samples_path)
    else:
        samples_path = './work_dirs/temp_samples'
        suffix = 1
        while osp.exists(samples_path):
            samples_path = f'./work_dirs/temp_samples_{suffix}'
            suffix += 1
        os.makedirs(samples_path)

    mmcv.print_log(
        f'Sample {num_samples} images for whole-image export', 'mmgen')
    pbar = mmcv.ProgressBar(num_samples)

    fields = fields_to_export(field)
    selected = {name: [] for name in fields}
    for begin, data_batch in zip(
            range(0, num_samples, batch_size), data_loader):
        end = min(begin + batch_size, num_samples)
        output_dict = None
        if 'fake' in fields:
            output_dict = model(
                data_batch[f'img_{source_domain}'],
                test_mode=True,
                target_domain=target_domain)

        for name in fields:
            batch = field_batch(
                name, data_batch, output_dict, source_domain, target_domain)
            for image in batch:
                selected[name].append(
                    crop_edges(image[channel_list, ...], n_edge).unsqueeze(0))
        pbar.update(end - begin)

    print()
    for name in fields:
        max_norm = field_norm(name)[channel_list]
        img_list = torch.cat(selected[name], dim=0).detach().cpu()
        tiff_image = (
            torch.cat(selected[name], dim=2).detach().squeeze(0).cpu().numpy())
        io.imsave(
            osp.join(samples_path, output_name(name, checkpoint_name, 'tiff')),
            tiff_image)
        save_image(
            zero_one_rgb(img_list, max_norm),
            osp.join(samples_path, output_name(name, checkpoint_name, 'png')),
            nrow=9,
            padding=0,
            scale_each=False)


def main():
    args = parse_args()
    if args.eval != ['none']:
        raise NotImplementedError(
            'This project keeps only the image export path.')
    if not args.generate_whole:
        raise NotImplementedError(
            'test.sh uses --generate_whole; other modes were removed.')

    cfg = Config.fromfile(args.config)
    checkpoint_name = (
        args.checkpoint_name or osp.splitext(osp.basename(args.checkpoint))[0])
    dirname = osp.dirname(args.checkpoint)
    ckpt = osp.basename(args.checkpoint)
    log_path = None if 'http' in args.checkpoint else osp.join(
        dirname, f'{osp.splitext(ckpt)[0]}_eval_log.txt')
    logger = get_root_logger(
        log_file=log_path, log_level=cfg.log_level, file_mode='a')
    logger.info('evaluation')

    if (args.delete_pre_folder and args.samples_path
            and osp.exists(args.samples_path)):
        shutil.rmtree(args.samples_path)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    assert isinstance(model, BaseTranslationModel)
    model.eval()
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])

    target_domain = args.target_domain or model.module._default_domain
    source_domain = model.module.get_other_domains(target_domain)[0]
    num_samples = count_test_images(cfg)
    print('number of demo images: ', num_samples)
    assert num_samples > 0

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfg.data.get('val_workers_per_gpu',
                                     cfg.data.workers_per_gpu),
        dist=False,
        shuffle=False)
    generate_whole_image(
        model,
        data_loader,
        args.samples_path,
        checkpoint_name,
        source_domain,
        target_domain,
        args.batch_size,
        num_samples,
        args.channel_list,
        args.n_edge,
        args.field)


if __name__ == '__main__':
    main()
