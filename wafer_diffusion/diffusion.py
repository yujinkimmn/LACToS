"""DDPM training and full-chain DPS for y=x+sigma*n; no shortened-chain SDEdit substitution."""
from __future__ import annotations

import math
import torch


class Diffusion:
    def __init__(self, steps=1000, schedule="linear", device="cpu"):
        if not 8 <= steps <= 4000:
            raise ValueError("Diffusion steps must be between 8 and 4000.")
        if schedule == "linear":
            beta = torch.linspace(1e-4, .02, steps, dtype=torch.float64)
        elif schedule == "cosine":
            t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
            abar = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
            abar = abar / abar[0]
            beta = (1 - abar[1:] / abar[:-1]).clamp(1e-8, .999)
        else:
            raise ValueError("Unknown noise schedule.")
        alpha = 1 - beta
        abar = alpha.cumprod(0)
        previous = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
        self.steps = steps
        self.abar = abar.float().to(device)
        self.coef_x0 = (beta * previous.sqrt() / (1 - abar)).float().to(device)
        self.coef_xt = (alpha.sqrt() * (1 - previous) / (1 - abar)).float().to(device)
        self.variance = (beta * (1 - previous) / (1 - abar)).float().to(device)

    def q_sample(self, clean, time, noise):
        abar = self.abar[time].view(-1, 1, 1, 1)
        return abar.sqrt() * clean + (1 - abar).sqrt() * noise

    def reconstruct(self, model, clean, draws, sigma, scale, seed, progress=None):
        if clean.shape[0] != 1 or not 1 <= draws <= 64:
            raise ValueError("Sampling expects one input image and 1..64 draws.")
        if not 0 < sigma <= 5 or not 0 <= scale <= 5:
            raise ValueError("Invalid measurement noise sigma or DPS scale.")
        model.eval()
        results = []
        # Each draw has its own RNG for measurement, initial state AND every reverse step.
        # Sequential draws bound GPU memory and make restart/draw-count changes reproducible.
        for draw in range(draws):
            generator = torch.Generator(device=clean.device).manual_seed((seed + draw * 1_000_003) % (2**63 - 1))
            def noise():
                return torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
            measurement = clean + sigma * noise()
            image = noise()
            for index in reversed(range(self.steps)):
                with torch.enable_grad():
                    image = image.detach().requires_grad_(True)
                    time = torch.full((1,), index, device=clean.device, dtype=torch.long)
                    eps = model(image, time)
                    abar = self.abar[index]
                    x0 = ((image - (1 - abar).sqrt() * eps) / abar.sqrt()).clamp(-1, 1)
                    mean = self.coef_x0[index] * x0 + self.coef_xt[index] * image
                    if scale:
                        residual = (measurement - x0).flatten(1).norm(dim=1).sum()
                        gradient = torch.autograd.grad(residual, image)[0]
                    else:
                        gradient = torch.zeros_like(image)
                image = mean.detach() + (self.variance[index].sqrt() * noise() if index else 0) - scale * gradient
                if not torch.isfinite(image).all():
                    raise FloatingPointError("Non-finite DPS sample; check training and guidance settings.")
            results.append(image.detach().clamp(-1, 1).cpu())
            if progress:
                progress(draw + 1, draws)
        return torch.cat(results)
