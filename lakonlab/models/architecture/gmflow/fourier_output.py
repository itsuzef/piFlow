"""FourierOutput2D — transformer head for the Fourier policy.

Produces the two tensors consumed by `FourierPolicy`:
    • x_hat_0           — predicted clean sample, (B, C, H, W).
    • fourier_sin_coeffs — sin-series coefficients a_m(ξ), (B, M, C, H, W).

Designed as a pure-reshape head mirroring the structure of GMOutput2D.
No learnable parameters in v1; the spatial projection lives in the
transformer backbone, not in this module.

Scope v1:
    • 2D only — image-shape spatial axes (H, W). 5D/video deferred.
    • Pure reshape — no Linear, no activation, no logstd MLP.
    • Schedule-locked first cut: FourierPolicy assumes α = 1 − σ
      (see fourier.py schedule note), so cross-schedule deployment
      requires a coordinated change in both files.
"""

import torch
import torch.nn as nn

from dataclasses import dataclass
from diffusers.utils import BaseOutput


@dataclass
class FourierModelOutput(BaseOutput):
    """
    The output of FourierOutput2D.

    Args:
        x_hat_0 (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Predicted clean sample x̂_0.
        fourier_sin_coeffs (`torch.Tensor` of shape
        `(batch_size, num_harmonics, num_channels, height, width)`):
            Sin-series coefficients a_m(ξ) for m = 1..M.
    """

    x_hat_0: torch.Tensor
    fourier_sin_coeffs: torch.Tensor


class FourierOutput2D(nn.Module):

    def __init__(self,
                 num_harmonics,
                 out_channels,
                 embed_dim=None):
        super(FourierOutput2D, self).__init__()
        self.M = num_harmonics
        self.out_channels = out_channels
        # `embed_dim` is reserved for a future emb-driven branch (e.g. an
        # adaptive scalar gain on the Fourier residual). Accepted but unused
        # in v1 to keep constructor parity with GMOutput2D.
        self.embed_dim = embed_dim

    def init_weights(self):
        # No-op in v1: this head holds no learnable parameters. Method is
        # kept for interface parity with GMOutput2D so the transformer's
        # init_weights pass can call it uniformly.
        pass

    def forward(self, x, emb=None):
        bs, _, h, w = x.size()
        x_hat_flat, coeffs_flat = x.split(
            [self.out_channels, self.M * self.out_channels], dim=1)
        fourier_sin_coeffs = coeffs_flat.view(bs, self.M, self.out_channels, h, w)
        return FourierModelOutput(
            x_hat_0=x_hat_flat,
            fourier_sin_coeffs=fourier_sin_coeffs)
