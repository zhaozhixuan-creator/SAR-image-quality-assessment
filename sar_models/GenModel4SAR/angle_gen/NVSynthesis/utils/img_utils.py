import math

import numpy as np
import torch


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert a tensor image to a NumPy image array."""
    tensor = tensor.squeeze().detach().float().cpu().clamp_(*min_max)

    if tensor.dim() == 4:
        rows = []
        nrow = int(math.ceil(math.sqrt(tensor.size(0))))
        for row_start in range(0, tensor.size(0), nrow):
            cells = [_single_tensor_to_image(x) for x in tensor[row_start:row_start + nrow]]
            cells.extend(torch.zeros_like(cells[0]) for _ in range(nrow - len(cells)))
            row = torch.cat(cells, dim=1)
            rows.append(row)
        img_np = torch.cat(rows, dim=0).numpy()
    else:
        img_np = _single_tensor_to_image(tensor).numpy()

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)


def _single_tensor_to_image(tensor):
    if tensor.dim() == 2:
        return tensor
    if tensor.dim() == 3 and tensor.size(0) == 3:
        return tensor[[2, 1, 0]].permute(1, 2, 0)
    if tensor.dim() == 3:
        return torch.sqrt(torch.sum(torch.pow(tensor, 2), dim=0) / 2)
    raise TypeError(
        "Only support 4D, 3D and 2D tensor. "
        "But received with dimension: {:d}".format(tensor.dim())
    )
