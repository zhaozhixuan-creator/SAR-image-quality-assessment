"""
Per-class batch comparison:
- For each class, sample up to N real images (default 100) from the dataset.
- Use their labels to condition G and generate images.
- Compute SSIM per pair (grayscale).
- Pick top K per class (default 4).
- Report mean/std SSIM overall and save CSV.
- Save a collage of all selected pairs: 7 columns (classes), 4 rows (top-K), left=real, right=gen, with spacing.
"""

import argparse
import math
import os
import random
import csv
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

import dnnlib
import legacy
from training import dataset

DEFAULT_CLASS_NAMES = ["2S1", "BRDM_2", "D7", "T62", "ZIL131", "ZSU_23_4", "SLICY"]


def load_dataset(path):
    return dataset.ImageFolderDataset(path=path, use_labels=True, max_size=None, xflip=False)


def tensor_to_uint8(img_t):
    img = (img_t.clamp(-1, 1).add(1).mul(127.5).permute(0, 2, 3, 1).cpu().numpy())[0]
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    return img


def chw_to_hwc_uint8(img_chw):
    img = img_chw.transpose(1, 2, 0).astype(np.uint8)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    return img


def center_crop_ratio(img, ratio):
    h, w = img.shape[:2]
    ratio = max(0.1, min(1.0, float(ratio)))
    new_h = max(1, int(round(h * ratio)))
    new_w = max(1, int(round(w * ratio)))
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    return img[y0:y0 + new_h, x0:x0 + new_w]


def circ_dist(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def pick_spread_samples(items, k):
    if len(items) <= k:
        return items
    angles = [it.get('angle_deg', 0.0) for it in items]
    # If angles are all the same, fall back to random.
    if max(angles) - min(angles) < 1e-6:
        return random.sample(items, k)

    chosen = []
    remaining = list(range(len(items)))
    start_idx = random.choice(remaining)
    chosen.append(start_idx)
    remaining.remove(start_idx)

    while len(chosen) < k and remaining:
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            ang = angles[idx]
            min_dist = min(circ_dist(ang, angles[c]) for c in chosen)
            if min_dist > best_score:
                best_score = min_dist
                best_idx = idx
        chosen.append(best_idx)
        remaining.remove(best_idx)

    return [items[i] for i in chosen]

def make_collage(per_class_pairs, save_path, classes=7, topk=4, spacing=6, class_names=None):
    """
    per_class_pairs: list length=classes, each list of dicts with keys real, gen, cls, angle_deg.
    Layout: columns=classes, rows=topk (4x7=28 pairs), each cell is Real|Gen with spacing.
    Left = Real, Right = Fake.
    """
    # Filter empty classes
    valid_classes = [p for p in per_class_pairs if p]
    if not valid_classes:
        return
    h, w = valid_classes[0][0]['real'].shape[:2]
    pair_w, pair_h = 2 * w, h
    cols = classes
    rows = topk
    font = ImageFont.load_default()
    margin = 4
    class_names = class_names or []

    font_h = font.getbbox("Real")[3] - font.getbbox("Real")[1]
    header_h = font_h + 2 * spacing  # Reserve enough space for headers.

    canvas_w = cols * pair_w + spacing * (cols + 1)
    canvas_h = header_h + rows * pair_h + spacing * (rows + 1)
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
    draw_canvas = ImageDraw.Draw(canvas)

    # Header: Real/Fake labels (slightly lower).
    header_y = spacing
    for c in range(classes):
        x0 = spacing + c * (pair_w + spacing)
        real_label_pos = (x0 + w * 0.25, header_y)
        fake_label_pos = (x0 + w + w * 0.25, header_y)
        draw_canvas.text(real_label_pos, "Real", fill=(0, 0, 0), font=font)
        draw_canvas.text(fake_label_pos, "Fake", fill=(0, 0, 0), font=font)

    for c in range(classes):
        pairs = per_class_pairs[c] if c < len(per_class_pairs) else []
        for r in range(rows):
            if r >= len(pairs):
                continue
            p = pairs[r]
            r_arr = p['real']
            g_arr = p['gen']
            r_img = Image.fromarray(r_arr if r_arr.ndim == 3 else np.stack([r_arr] * 3, axis=2))
            g_img = Image.fromarray(g_arr if g_arr.ndim == 3 else np.stack([g_arr] * 3, axis=2))
            cell = Image.new('RGB', (pair_w, pair_h), color=(0, 0, 0))
            cell.paste(r_img, (0, 0))      # Left: Real
            cell.paste(g_img, (w, 0))      # Right: Fake
            draw = ImageDraw.Draw(cell)
            cls_name = class_names[p['cls']] if p['cls'] < len(class_names) else f"C{p['cls']}"
            text = f"{cls_name} ang {p['angle_deg']:.1f} deg"
            draw.text((margin, margin), text, fill=(255, 255, 0), font=font)

            x0 = spacing + c * (pair_w + spacing)
            y0 = header_h + spacing + r * (pair_h + spacing)
            canvas.paste(cell, (x0, y0))
    canvas.save(save_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--network', required=True, help='Path to network-snapshot-*.pkl')
    p.add_argument('--data', required=True, help='Dataset zip/dir (with labels)')
    p.add_argument('--per-class', type=int, default=100, help='Samples per class to evaluate')
    p.add_argument('--topk', type=int, default=4, help='Top-K per class for collage')
    p.add_argument('--classes', type=int, default=7, help='Number of classes (one-hot length)')
    p.add_argument('--ssim-crop-ratio', type=float, default=None,
                   help='Center-crop ratio before SSIM (e.g., 0.5 for half)')
    p.add_argument('--random-show-spread', action='store_true',
                   help='Show random samples per class with angles spread out')
    p.add_argument('--seed', type=int, default=0, help='Random seed for sampling and z')
    p.add_argument('--outdir', default='batch_compare_out', help='Output directory')
    p.add_argument('--aasg-enabled', type=int, choices=[0, 1], default=1,
                   help='Enable AASG branch in generator forward (1=on, 0=off)')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.network, 'rb') as f:
        net = legacy.load_network_pkl(f)['G_ema'].to(device)

    # Load dataset
    dset = load_dataset(args.data)
    labels = getattr(dset, '_raw_labels', None)
    if labels is None:
        labels = dset._get_raw_labels() if hasattr(dset, '_get_raw_labels') else None
    if labels is None or labels.shape[1] == 0:
        raise RuntimeError('Dataset has no labels; repackage with labels!=None.')
    labels = torch.from_numpy(labels)
    class_ids = labels[:, :args.classes].argmax(dim=1)

    # Prepare index lists per class
    per_class_indices = []
    for c in range(args.classes):
        idxs = (class_ids == c).nonzero(as_tuple=False).view(-1).tolist()
        random.shuffle(idxs)
        per_class_indices.append(idxs[:args.per_class])

    results = []
    per_class_results = [[] for _ in range(args.classes)]
    for c in range(args.classes):
        for idx in per_class_indices[c]:
            real_img_chw, real_label_np = dset[idx]
            real_img = chw_to_hwc_uint8(real_img_chw)
            angle_deg = None
            angle_idx = args.classes
            if real_label_np.shape[0] >= angle_idx + 2:
                angle_rad = math.atan2(real_label_np[angle_idx], real_label_np[angle_idx + 1])
                angle_deg = (math.degrees(angle_rad) % 360.0)

            label = torch.from_numpy(real_label_np).to(device)
            z = torch.randn([1, net.z_dim], device=device)
            cvec = label.unsqueeze(0).to(torch.float32)
            with torch.no_grad():
                gen = net(z, cvec, noise_mode='const', aasg_enabled=bool(args.aasg_enabled))
            gen_img = tensor_to_uint8(gen)

            real_gray = real_img if real_img.ndim == 2 else np.array(Image.fromarray(real_img).convert('L'))
            gen_gray = gen_img if gen_img.ndim == 2 else np.array(Image.fromarray(gen_img).convert('L'))
            if args.ssim_crop_ratio is not None:
                real_gray = center_crop_ratio(real_gray, args.ssim_crop_ratio)
                gen_gray = center_crop_ratio(gen_gray, args.ssim_crop_ratio)
            ssim_val = ssim(real_gray, gen_gray, data_range=255)

            rec = dict(idx=idx, cls=c, angle_deg=angle_deg if angle_deg is not None else 0.0,
                       ssim=ssim_val, real=real_img, gen=gen_img)
            results.append(rec)
            per_class_results[c].append(rec)

    # Stats
    ssim_vals = [r['ssim'] for r in results]
    mean_ssim = float(np.mean(ssim_vals))
    std_ssim = float(np.std(ssim_vals))
    with open(os.path.join(args.outdir, 'stats.txt'), 'w') as f:
        f.write(f'SSIM mean: {mean_ssim:.4f}\n')
        f.write(f'SSIM std: {std_ssim:.4f}\n')
    with open(os.path.join(args.outdir, 'results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['idx', 'cls', 'angle_deg', 'ssim'])
        writer.writeheader()
        for r in results:
            writer.writerow({'idx': r['idx'], 'cls': r['cls'], 'angle_deg': r['angle_deg'], 'ssim': r['ssim']})

    # Select top-K per class
    top_per_class = []
    for c in range(args.classes):
        sorted_c = sorted(per_class_results[c], key=lambda x: x['ssim'], reverse=True)
        if args.random_show_spread:
            top_per_class.append(pick_spread_samples(sorted_c, min(args.topk, len(sorted_c))))
        else:
            top_per_class.append(sorted_c[:min(args.topk, len(sorted_c))])

    # Collage
    collage_path = os.path.join(args.outdir, 'top_pairs.png')
    # Prepare class names
    class_names = DEFAULT_CLASS_NAMES[:args.classes]
    if len(class_names) < args.classes:
        class_names += [f"C{i}" for i in range(len(class_names), args.classes)]
    make_collage(top_per_class, collage_path, classes=args.classes, topk=args.topk, spacing=6, class_names=class_names)

    print(f'Done. SSIM mean={mean_ssim:.4f}, std={std_ssim:.4f}')
    print(f'Saved stats/results to {args.outdir}')


if __name__ == '__main__':
    main()
