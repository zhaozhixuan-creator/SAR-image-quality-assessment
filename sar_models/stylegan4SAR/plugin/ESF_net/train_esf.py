import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .ESF_config import DEFAULT_ESF_CONFIG, ESFConfig
from .esf_dataset import ASCZipDataset
from .esf_model import ESFAttributeNet


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sort_points_by_xy(t: torch.Tensor) -> torch.Tensor:
    key = t[:, :, 0] + 0.01 * t[:, :, 1]
    idx = torch.argsort(key, dim=1)
    idx_expand = idx.unsqueeze(-1).expand(-1, -1, t.shape[2])
    return torch.gather(t, 1, idx_expand)


def _set_regression_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, cfg: ESFConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pred_s = _sort_points_by_xy(pred)
    target_s = _sort_points_by_xy(target)

    # 位置误差放大: 提高小图像下 xy 对梯度的贡献。
    xy_scale = float(cfg.xy_error_scale)
    pred_xy = pred_s[:, :, 0:2] * xy_scale
    target_xy = target_s[:, :, 0:2] * xy_scale
    pos_loss = F.smooth_l1_loss(pred_xy, target_xy, reduction="none")
    pos_loss = (pos_loss * mask).sum() / ((mask.sum() * 2).clamp(min=1.0))

    amp_loss = F.smooth_l1_loss(pred_s[:, :, 2:3], target_s[:, :, 2:3], reduction="none")
    amp_loss = (amp_loss * mask).sum() / (mask.sum().clamp(min=1.0))

    if pred_s.shape[2] > 3:
        alpha_loss = F.smooth_l1_loss(pred_s[:, :, 3:4], target_s[:, :, 3:4], reduction="none")
        alpha_loss = (alpha_loss * mask).sum() / (mask.sum().clamp(min=1.0))
    else:
        alpha_loss = torch.zeros([], device=pred.device)

    amp_dist = (pred_s[:, :, 2] - target_s[:, :, 2]).abs().mean()
    total = (
        cfg.xy_loss_weight * pos_loss
        + cfg.amp_loss_weight * amp_loss
        + cfg.alpha_loss_weight * alpha_loss
        + 0.2 * amp_dist
    )
    comps = {
        "loss_pos": pos_loss,
        "loss_amp": amp_loss,
        "loss_alpha": alpha_loss,
        "loss_amp_dist": amp_dist,
    }
    return total, comps


def _masked_attr_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    err = (pred - target).abs() * mask
    denom = mask.sum().clamp(min=1.0)
    mae_x = err[:, :, 0].sum() / denom
    mae_y = err[:, :, 1].sum() / denom
    mae_amp = err[:, :, 2].sum() / denom if pred.shape[2] > 2 else torch.zeros([], device=pred.device)
    mae_alpha = err[:, :, 3].sum() / denom if pred.shape[2] > 3 else torch.zeros([], device=pred.device)
    return {"mae_x": mae_x, "mae_y": mae_y, "mae_amp": mae_amp, "mae_alpha": mae_alpha}


def _run_epoch(
    model: ESFAttributeNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: ESFConfig,
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_mae_x = 0.0
    total_mae_y = 0.0
    total_mae_amp = 0.0
    total_mae_alpha = 0.0
    total_grad_norm = 0.0
    n = 0
    for x, target, mask, _meta in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss_raw, _ = _set_regression_loss(pred=pred, target=target, mask=mask, cfg=cfg)
        loss = loss_raw * float(cfg.loss_scale)

        if train:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip_norm))
            optimizer.step()
            total_grad_norm += float(grad_norm.detach().cpu())

        maes = _masked_attr_mae(pred=pred, target=target, mask=mask)
        total_loss += float(loss_raw.detach().cpu())
        total_mae_x += float(maes["mae_x"].detach().cpu())
        total_mae_y += float(maes["mae_y"].detach().cpu())
        total_mae_amp += float(maes["mae_amp"].detach().cpu())
        total_mae_alpha += float(maes["mae_alpha"].detach().cpu())
        n += 1

    denom = max(n, 1)
    return {
        "loss": total_loss / denom,
        "mae_x": total_mae_x / denom,
        "mae_y": total_mae_y / denom,
        "mae_amp": total_mae_amp / denom,
        "mae_alpha": total_mae_alpha / denom,
        "grad_norm": (total_grad_norm / denom) if train else 0.0,
    }


def _norm_to_px(v: float, size: int) -> int:
    x = ((float(v) + 1.0) * 0.5) * size - 0.5
    return int(round(max(0.0, min(size - 1.0, x))))


@torch.no_grad()
def _save_epoch_viz(model: ESFAttributeNet, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, out_path: Path, device: torch.device) -> None:
    model.eval()
    pred = model(x.unsqueeze(0).to(device))[0].detach().cpu()
    x_img = (x[0].detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
    h, w = x_img.shape

    base = Image.fromarray(x_img, mode="L").convert("RGB")
    gt_img = base.copy()
    pd_img = base.copy()
    draw_gt = ImageDraw.Draw(gt_img)
    draw_pd = ImageDraw.Draw(pd_img)

    for i in range(target.shape[0]):
        if float(mask[i, 0]) < 0.5:
            continue
        gx = _norm_to_px(float(target[i, 0]), w)
        gy = _norm_to_px(float(target[i, 1]), h)
        draw_gt.ellipse((gx - 3, gy - 3, gx + 3, gy + 3), outline=(0, 255, 0), width=1)

    for i in range(pred.shape[0]):
        px = _norm_to_px(float(pred[i, 0]), w)
        py = _norm_to_px(float(pred[i, 1]), h)
        draw_pd.ellipse((px - 3, py - 3, px + 3, py + 3), outline=(255, 0, 0), width=1)

    canvas = Image.new("RGB", (w * 2, h), color=(0, 0, 0))
    canvas.paste(gt_img, (0, 0))
    canvas.paste(pd_img, (w, 0))
    d = ImageDraw.Draw(canvas)
    d.text((4, 4), "GT", fill=(0, 255, 0))
    d.text((w + 4, 4), "Pred", fill=(255, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _split_dataset(dataset: Dataset, val_ratio: float) -> Tuple[Dataset, Dataset]:
    n = len(dataset)
    n_val = max(1, int(round(n * val_ratio)))
    n_train = max(1, n - n_val)
    if n_train + n_val != n:
        n_val = n - n_train
    return random_split(dataset, [n_train, n_val])


def _make_folds(n: int, k: int) -> List[np.ndarray]:
    idx = np.arange(n)
    np.random.shuffle(idx)
    return [arr.astype(np.int64) for arr in np.array_split(idx, k)]


def _train_one_fold(
    fold_id: int,
    train_set: Dataset,
    val_set: Dataset,
    cfg: ESFConfig,
    device: torch.device,
    outdir: Path,
) -> Dict[str, float]:
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = ESFAttributeNet(
        k_points=cfg.k_points,
        attr_dim=cfg.attr_dim,
        base_channels=cfg.base_channels,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(cfg.lr_scheduler_tmax), 1),
        eta_min=float(cfg.lr_scheduler_min),
    )

    best_val = float("inf")
    history = []
    fold_dir = outdir / f"fold_{fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "esf_attr_best.pt"
    last_path = fold_dir / "esf_attr_last.pt"
    viz_dir = fold_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, cfg.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, cfg, train=True)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, optimizer, device, cfg, train=False)
        history.append({"epoch": ep, "train": train_metrics, "val": val_metrics})
        print(
            f"[ESF][F{fold_id:02d}] epoch={ep:03d} "
            f"train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f} "
            f"val_mae(x,y,A,a)=({val_metrics['mae_x']:.4f},{val_metrics['mae_y']:.4f},{val_metrics['mae_amp']:.4f},{val_metrics['mae_alpha']:.4f}) "
            f"grad_norm={train_metrics['grad_norm']:.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if cfg.viz_interval > 0 and ep % cfg.viz_interval == 0:
            ridx = random.randrange(len(val_set))
            vx, vt, vm, _ = val_set[ridx]
            _save_epoch_viz(model=model, x=vx, target=vt, mask=vm, out_path=viz_dir / f"epoch_{ep:03d}.png", device=device)

        payload = {
            "model": model.state_dict(),
            "config": {
                "k_points": cfg.k_points,
                "attr_dim": cfg.attr_dim,
                "base_channels": cfg.base_channels,
                "dropout": cfg.dropout,
            },
            "fold_id": fold_id,
            "epoch": ep,
            "val_loss": val_metrics["loss"],
        }
        torch.save(payload, str(last_path))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(payload, str(best_path))
        scheduler.step()

    (fold_dir / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"best_val": best_val, "best_ckpt": str(best_path).replace("\\", "/"), "last_ckpt": str(last_path).replace("\\", "/")}


def train(cfg: ESFConfig = DEFAULT_ESF_CONFIG) -> Dict[str, object]:
    if int(cfg.seed) >= 0:
        _set_seed(int(cfg.seed))
        print(f"[ESF] fixed_seed={int(cfg.seed)}")
    else:
        print("[ESF] seed=random (not fixed)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = ASCZipDataset(zip_path=cfg.dataset_zip, k_points=cfg.k_points, attr_dim=cfg.attr_dim)

    fold_results = []
    if int(cfg.k_folds) > 1:
        folds = _make_folds(len(dataset), int(cfg.k_folds))
        for i in range(len(folds)):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i], axis=0)
            train_set = Subset(dataset, train_idx.tolist())
            val_set = Subset(dataset, val_idx.tolist())
            fr = _train_one_fold(i + 1, train_set, val_set, cfg, device, outdir)
            fr.update({"fold": i + 1, "train_size": len(train_set), "val_size": len(val_set)})
            fold_results.append(fr)
        best_val = float(min(fr["best_val"] for fr in fold_results))
        best_ckpt = [fr["best_ckpt"] for fr in fold_results if fr["best_val"] == best_val][0]
        last_ckpt = fold_results[-1]["last_ckpt"]
    else:
        train_set, val_set = _split_dataset(dataset, cfg.val_ratio)
        fr = _train_one_fold(1, train_set, val_set, cfg, device, outdir)
        fr.update({"fold": 1, "train_size": len(train_set), "val_size": len(val_set)})
        fold_results.append(fr)
        best_val = float(fr["best_val"])
        best_ckpt = fr["best_ckpt"]
        last_ckpt = fr["last_ckpt"]

    info = {
        "best_ckpt": best_ckpt,
        "last_ckpt": last_ckpt,
        "dataset_zip": str(cfg.dataset_zip).replace("\\", "/"),
        "num_samples": len(dataset),
        "k_folds": int(cfg.k_folds),
        "best_val": best_val,
        "fold_results": fold_results,
    }
    cfg_json = {k: (str(v).replace("\\", "/") if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    (outdir / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "train_config.json").write_text(json.dumps(cfg_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def main() -> None:
    info = train(DEFAULT_ESF_CONFIG)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

