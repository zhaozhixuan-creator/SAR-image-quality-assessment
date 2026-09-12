from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from .esf_model import ESFAttributeNet


class FrozenESFAdapter(nn.Module):
    """Frozen attribute predictor for future discriminator injection."""

    def __init__(self, model: ESFAttributeNet, embed_dim: int = 128) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        in_dim = self.model.k_points * self.model.attr_dim
        self.projector = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
        )

    @torch.no_grad()
    def predict_attrs(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            attrs = self.model(x)  # [B,K,D]
        emb = self.projector(attrs.flatten(1))
        return {"attrs": attrs, "embedding": emb}


def build_frozen_esf_adapter(ckpt_path: Path, device: torch.device, embed_dim: int = 128) -> FrozenESFAdapter:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model = ESFAttributeNet(
        k_points=int(ckpt["config"]["k_points"]),
        attr_dim=int(ckpt["config"]["attr_dim"]),
        base_channels=int(ckpt["config"]["base_channels"]),
        dropout=float(ckpt["config"]["dropout"]),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    adapter = FrozenESFAdapter(model=model, embed_dim=embed_dim).to(device)
    return adapter

