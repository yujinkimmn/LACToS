"""Small epsilon-predicting U-Net, implemented here without upstream model code/weights."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    return nn.GroupNorm(math.gcd(channels, 8), channels)


class Residual(nn.Module):
    def __init__(self, incoming, outgoing, time_dim):
        super().__init__()
        self.first = nn.Sequential(norm(incoming), nn.SiLU(), nn.Conv2d(incoming, outgoing, 3, padding=1))
        self.time = nn.Linear(time_dim, outgoing)
        self.second = nn.Sequential(norm(outgoing), nn.SiLU(), nn.Conv2d(outgoing, outgoing, 3, padding=1))
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, image, time):
        hidden = self.first(image) + self.time(F.silu(time))[:, :, None, None]
        return self.skip(image) + self.second(hidden)


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = norm(channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.output = nn.Conv1d(channels, channels, 1)

    def forward(self, image):
        b, c, h, w = image.shape
        q, k, v = self.qkv(self.norm(image).reshape(b, c, h * w)).chunk(3, dim=1)
        # math SDPA keeps the second-order/input gradient used by DPS available.
        weights = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(c), dim=-1)
        result = torch.bmm(v, weights.transpose(1, 2))
        return image + self.output(result).reshape(b, c, h, w)


class WaferUNet(nn.Module):
    def __init__(self, base_channels=32, channel_mult=(1, 2, 4, 4), attention=True):
        super().__init__()
        widths = [base_channels * m for m in channel_mult]
        self.time_width = base_channels
        tdim = base_channels * 4
        self.time = nn.Sequential(nn.Linear(base_channels, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.input = nn.Conv2d(3, widths[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        current = widths[0]
        for i, width in enumerate(widths):
            self.down.append(Residual(current, width, tdim))
            current = width
            if i < len(widths) - 1:
                self.downsample.append(nn.Conv2d(width, width, 3, stride=2, padding=1))
        self.mid1 = Residual(current, current, tdim)
        self.attention = Attention(current) if attention else nn.Identity()
        self.mid2 = Residual(current, current, tdim)
        self.up = nn.ModuleList()
        for width in reversed(widths):
            self.up.append(Residual(current + width, width, tdim))
            current = width
        self.output = nn.Sequential(norm(current), nn.SiLU(), nn.Conv2d(current, 3, 3, padding=1))

    def forward(self, image, time):
        half = self.time_width // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=image.device) / max(half - 1, 1))
        angles = time.float()[:, None] * frequencies[None]
        time = self.time(torch.cat([angles.sin(), angles.cos()], dim=1))
        hidden, skips = self.input(image), []
        for i, block in enumerate(self.down):
            hidden = block(hidden, time)
            skips.append(hidden)
            if i < len(self.downsample):
                hidden = self.downsample[i](hidden)
        hidden = self.mid2(self.attention(self.mid1(hidden, time)), time)
        for block, skip in zip(self.up, reversed(skips)):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = block(torch.cat([hidden, skip], dim=1), time)
        return self.output(hidden)


def build_model(config):
    return WaferUNet(config["base_channels"], tuple(config["channel_mult"]), config["attention"])
