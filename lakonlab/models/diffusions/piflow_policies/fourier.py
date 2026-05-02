# Copyright (c) 2026 youssefhemimy / NOCM-piFlow
#
# Fourier policy — boundary-pinned Fourier curve between anchors
# (x_t_src, x̂_0), with a sin-series correction term.
#
# Schedule note:
# This implementation is locked to the linear (VE) noise schedule
# alpha = 1 − sigma. Two aspects of the formulation depend on this:
#   1. The boundary basis φ_0(τ)=1−τ, φ_1(τ)=τ achieves zero geometric
#      residual at perfect prediction only under the linear schedule. Under
#      a non-linear (e.g. cosine VP) schedule the boundary basis absorbs a
#      residual that the network's Fourier coefficients must compensate.
#   2. The F2 velocity u = ∂_σ f uses dτ/dσ = −1/σ_t_src derived from
#      τ(σ) = 1 − σ/σ_t_src; the τ-axis is schedule-agnostic, but the
#      zero-residual guarantee at the boundary is schedule-dependent.
# Supporting a non-linear schedule requires re-deriving the boundary basis
# for VP and updating the coefficient distribution accordingly.

import math

import torch

from typing import Dict

from .base import BasePolicy


class FourierPolicy(BasePolicy):
    """Fourier policy. Represents the diffusion path between anchors as a
    boundary-pinned Fourier curve in τ ∈ [0, 1].

    Path (linear-boundary basis + sin-only residual, Option A anchors):

        f(τ, ξ) = (1 − τ)·x_t_src(ξ) + τ·x̂_0(ξ)
                  + Σ_{m=1..M} a_m(ξ) · sin(m π τ)

    where τ(σ) = 1 − σ/σ_t_src, x_t_src is the cached anchor sample, and
    (x̂_0, {a_m}) come from the denoising network's head.

    The number of harmonics M is inferred from the denoising output.

    Args:
        denoising_output (dict): The output of the denoising model, with keys
            x_hat_0 (torch.Tensor): Predicted clean sample. Shape (B, C, H, W)
                or (B, C, T, H, W) — same as x_t_src.
            fourier_sin_coeffs (torch.Tensor): Sin-series coefficients
                a_m(ξ). Shape (B, M, C, H, W) or (B, M, C, T, H, W) — an
                extra M-axis at position 1.
        x_t_src (torch.Tensor): Anchor noisy sample. Shape (B, C, H, W) or
            (B, C, T, H, W).
        sigma_t_src (torch.Tensor): Anchor noise level. Shape (B,); reshaped
            for broadcast against x_t_src (same idiom as GMFlowPolicy).
        checkpointing (bool): Reserved — gradient checkpointing flag carried
            for parity with GMFlowPolicy. Not consumed yet (no posterior
            update path in f / df_dtau). Defaults to True.
        eps (float): Numerical floor. Defaults to 1e-6.
    """

    def __init__(
            self,
            denoising_output: Dict[str, torch.Tensor],
            x_t_src: torch.Tensor,
            sigma_t_src: torch.Tensor,
            checkpointing: bool = True,
            eps: float = 1e-6):
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.checkpointing = checkpointing
        self.eps = eps

        self.sigma_t_src = sigma_t_src.reshape(
            *sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))

        self.x_hat_0 = denoising_output['x_hat_0']
        self.fourier_sin_coeffs = denoising_output['fourier_sin_coeffs']
        self.M = self.fourier_sin_coeffs.size(1)

    def _reshape_tau(self, tau):
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, device=self.x_t_src.device, dtype=self.x_t_src.dtype)
        if tau.dim() == 0:
            tau = tau.expand(self.x_t_src.size(0))
        return tau.reshape(*tau.size(), *((self.ndim - tau.dim()) * [1]))

    def _m_grid(self):
        # m = (1, 2, ..., M) reshaped to broadcast against fourier_sin_coeffs:
        # axis 0 = batch (size 1), axis 1 = M, remaining axes = trailing 1s
        # so m * tau.unsqueeze(1) yields (B, M, 1, ..., 1).
        m = torch.arange(
            1, self.M + 1,
            device=self.fourier_sin_coeffs.device,
            dtype=self.fourier_sin_coeffs.dtype)
        return m.reshape(1, self.M, *((self.ndim - 1) * [1]))

    def f(self, tau):
        """Evaluate the Fourier path f(τ, ξ) at τ.

        Implements the linear-boundary + sin-only residual path under
        Option A anchors:

            f(τ) = (1 − τ)·x_t_src + τ·x̂_0 + Σ_m a_m · sin(m π τ)

        Args:
            tau (torch.Tensor or scalar): Path parameter, shape (B,) or
                scalar. Broadcast-promoted against x_t_src.

        Returns:
            torch.Tensor: f(τ), shape matching x_t_src — (B, C, H, W) or
            (B, C, T, H, W).
        """
        tau = self._reshape_tau(tau)
        boundary = (1 - tau) * self.x_t_src + tau * self.x_hat_0

        m = self._m_grid()
        tau_m = tau.unsqueeze(1)  # add M axis
        sin_mpi_tau = torch.sin(m * math.pi * tau_m)
        correction = (self.fourier_sin_coeffs * sin_mpi_tau).sum(dim=1)

        return boundary + correction

    def df_dtau(self, tau):
        """Analytic τ-derivative of the Fourier path.

            ∂f/∂τ = (x̂_0 − x_t_src) + Σ_m a_m · m π · cos(m π τ)

        Composing with dτ/dσ = −1/σ_t_src gives the F2 velocity
        u = ∂_σ f = −(1/σ_t_src) · ∂_τ f — used by the `pi` method.

        Args:
            tau (torch.Tensor or scalar): Path parameter, shape (B,) or
                scalar.

        Returns:
            torch.Tensor: ∂f/∂τ, shape matching x_t_src.
        """
        tau = self._reshape_tau(tau)
        boundary_deriv = self.x_hat_0 - self.x_t_src

        m = self._m_grid()
        tau_m = tau.unsqueeze(1)
        cos_mpi_tau = torch.cos(m * math.pi * tau_m)
        correction_deriv = (self.fourier_sin_coeffs * (m * math.pi) * cos_mpi_tau).sum(dim=1)

        return boundary_deriv + correction_deriv

    def pi(self, x_t, sigma_t):
        """Compute the F2 flow velocity u_t = ∂_σ f at the query σ_t.

        The Fourier path f(τ) is parametrised in τ(σ) = 1 − σ/σ_t_src; the
        F2 chain rule gives

            u = ∂_σ f = (∂_τ f) · (dτ/dσ) = −(1/σ_t_src) · df_dtau(τ).

        Note: the query x_t is not consumed — the Fourier curve is
        determined entirely by the cached anchors (x_t_src, x̂_0) and
        coefficients {a_m}; departures of the actual sample from the
        modelled curve are absorbed at the next denoising call, not here.
        The argument is kept for BasePolicy contract parity with
        GMFlowPolicy and DXPolicy.

        Args:
            x_t (torch.Tensor): Noisy input at time t. Shape matches
                x_t_src.
            sigma_t (torch.Tensor): Noise level at t. Shape (B,) or
                broadcast-compatible with x_t.

        Returns:
            torch.Tensor: Flow velocity u_t, shape matching x_t.
        """
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        sigma_t_src_safe = self.sigma_t_src.clamp(min=self.eps)
        tau = 1 - sigma_t / sigma_t_src_safe
        return self.df_dtau(tau) * (-1.0 / sigma_t_src_safe)

    def closed_form_step(self, sigma_t_dst):
        """Return f(τ_dst) — the closed-form destination state on the Fourier curve.

        Bypasses the Euler substep loop in policy_rollout. Valid because f(τ) is
        analytic; the substep loop exists for policies (GMFlow, DX) whose pi is
        not the derivative of a closed-form curve.

        Args:
            sigma_t_dst: destination noise level, shape (B,) or scalar.

        Returns:
            torch.Tensor: f(τ_dst), shape matching x_t_src.
        """
        if not isinstance(sigma_t_dst, torch.Tensor):
            sigma_t_dst = torch.tensor(
                sigma_t_dst, device=self.x_t_src.device, dtype=self.x_t_src.dtype)
        sigma_t_dst = sigma_t_dst.reshape(
            *sigma_t_dst.size(), *((self.ndim - sigma_t_dst.dim()) * [1]))
        tau_dst = 1 - sigma_t_dst / self.sigma_t_src.clamp(min=self.eps)
        return self.f(tau_dst)

    def copy(self):
        new_policy = self.__class__.__new__(self.__class__)
        new_policy.__dict__.update(self.__dict__)
        return new_policy

    def detach_(self):
        self.x_t_src = self.x_t_src.detach()
        self.x_hat_0 = self.x_hat_0.detach()
        self.fourier_sin_coeffs = self.fourier_sin_coeffs.detach()
        self.sigma_t_src = self.sigma_t_src.detach()
        return self

    def detach(self):
        new_policy = self.copy()
        return new_policy.detach_()
