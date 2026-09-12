import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, down: bool = True) -> None:
        super().__init__()
        stride = 2 if down else 1
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ESFAttributeNet(nn.Module):
    """Predict K ASC attributes from a SAR image."""

    def __init__(self, k_points: int = 8, attr_dim: int = 4, base_channels: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.k_points = int(k_points)
        self.attr_dim = int(attr_dim)
        c = int(base_channels)

        self.backbone = nn.Sequential(
            ConvBlock(1, c, down=True),       # 128 -> 64
            ConvBlock(c, c * 2, down=True),   # 64 -> 32
            ConvBlock(c * 2, c * 4, down=True),  # 32 -> 16
            ConvBlock(c * 4, c * 4, down=True),  # 16 -> 8
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, c * 8),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(c * 8, self.k_points * self.attr_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W] -> [B,K,D]
        feat = self.backbone(x)
        out = self.head(feat)
        out = out.view(x.shape[0], self.k_points, self.attr_dim)
        # 物理约束输出:
        # xy in [-1,1], A>=0 and sum(A)=1, alpha>=0
        xy = torch.tanh(out[:, :, 0:2])
        amp_logits = out[:, :, 2:3]
        amp = torch.softmax(amp_logits.squeeze(-1), dim=1).unsqueeze(-1)
        if self.attr_dim > 3:
            alpha = torch.nn.functional.softplus(out[:, :, 3:4])
            pred = torch.cat([xy, amp, alpha], dim=2)
        else:
            pred = torch.cat([xy, amp], dim=2)
        return pred
