# -*- coding: utf-8 -*-
"""
Deep Image Prior (DIP) for EHT imaging with:
- Student's-t losses (amplitude, closure phase) with correct phase wrapping
- TV and TSV regularizers
- L1 sparsity, flux constraint, compactness, inner-void, centroid penalties
- Gaussian initialization option
- MPS/CUDA/CPU compatible (float32)
- baseline-length reweighting for vis amps & closure phases
- optional learnable global image scale (alpha)
- coarse-to-fine Gaussian blur schedule on the image
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Device & dtype helpers
# ----------------------------
def pick_device(explicit: Optional[str] = None) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def default_dtype_for_device(dev: torch.device) -> torch.dtype:
    return torch.float32  # MPS doesn't support float64; float32 is fine for this pipeline.


# ----------------------------
# Utilities
# ----------------------------
def to_tensor(x, device, dtype):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def wrap_phase(x: torch.Tensor) -> torch.Tensor:
    """Wrap any angle to [-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def wrap_phase_diff(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Return wrapped (pred - obs) in [-pi, pi]."""
    return torch.atan2(torch.sin(pred - obs), torch.cos(pred - obs))


# ----------------------------
# ANSI color helpers for logging
# ----------------------------
_ANSI_COLORS = {
    "green": "\033[32m",
    "magenta": "\033[35m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
}
_ANSI_RESET = "\033[0m"


def _color_text(text: str, color: str) -> str:
    code = _ANSI_COLORS.get(color, "")
    return f"{code}{text}{_ANSI_RESET}" if code else text


def _tensor_item(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


# ----------------------------
# Regularizers
# ----------------------------
def tv_loss(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Isotropic TV on [B,1,H,W] (sum of gradient magnitudes)."""
    dx = img[..., :, 1:] - img[..., :, :-1]
    dy = img[..., 1:, :] - img[..., :-1, :]
    return torch.sum(torch.sqrt(dx * dx + eps)) + torch.sum(torch.sqrt(dy * dy + eps))


def tsv_loss(img: torch.Tensor) -> torch.Tensor:
    """Total Squared Variation (sum of squared gradients)."""
    dx = img[..., :, 1:] - img[..., :, :-1]
    dy = img[..., 1:, :] - img[..., :-1, :]
    return torch.sum(dx * dx) + torch.sum(dy * dy)


def l1_loss_mean(img: torch.Tensor) -> torch.Tensor:
    """Mean absolute intensity (sparsity)."""
    return torch.mean(torch.abs(img))


def centroid_penalty(img: torch.Tensor, device, dtype) -> torch.Tensor:
    """Flux-weighted centroid offset (pixels) from image center."""
    B, C, H, W = img.shape
    im2d = img[:, 0]
    x = (torch.arange(W, device=device, dtype=dtype) - (W // 2)).view(1, 1, W)
    y = (torch.arange(H, device=device, dtype=dtype) - (H // 2)).view(1, H, 1)
    total = torch.sum(im2d, dim=(1, 2)).clamp_min(1e-12)
    cx = torch.sum(im2d * x, dim=(1, 2)) / total
    cy = torch.sum(im2d * y, dim=(1, 2)) / total
    norm2 = (W / 2.0) ** 2 + (H / 2.0) ** 2
    return ((cx * cx + cy * cy) / (norm2 + 1e-12)).mean()


def radial_masks(H: int, W: int, cell_size_uas: float,
                 r_compact_uas: Optional[float], r_void_uas: Optional[float],
                 device, dtype):
    """Binary masks for compactness (outside Rc) and inner-void (inside R0)."""
    yy = (torch.arange(H, device=device, dtype=dtype) - H//2) * cell_size_uas
    xx = (torch.arange(W, device=device, dtype=dtype) - W//2) * cell_size_uas
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    R = torch.sqrt(X * X + Y * Y)
    mask_out = (R > float(r_compact_uas)).to(dtype) if (r_compact_uas and r_compact_uas > 0) else torch.zeros_like(R)
    mask_in  = (R < float(r_void_uas)).to(dtype)     if (r_void_uas and r_void_uas > 0)     else torch.zeros_like(R)
    return mask_out, mask_in


# ----------------------------
# Weighted Student t loss
# ----------------------------
class StudentTLoss1D(nn.Module):
    """
    Student's-t NLL for residuals r with optional per-point sigma and sample weights.
    Returns weighted mean by default (weights normalized by sum).
    """
    def __init__(self, nu: float = 3.0, learn_sigma: bool = False, init_sigma: float = 1.0):
        super().__init__()
        if nu <= 0:
            raise ValueError("nu must be > 0")
        self.nu = float(nu)
        self.learn_sigma = learn_sigma
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(init_sigma)))) if learn_sigma \
                         else nn.Parameter(torch.log(torch.tensor(float(init_sigma))), requires_grad=False)
        self.eps = 1e-8

    def forward(self, r: torch.Tensor, sigma: Optional[torch.Tensor] = None, weight: Optional[torch.Tensor] = None):
        # r: [N], sigma: [N] or scalar, weight: [N] or None
        if sigma is None:
            sigma2 = 1.0
        else:
            sigma2 = sigma * sigma
        s2 = torch.exp(2.0 * self.log_sigma)
        denom = self.nu * (s2 * (sigma2 if torch.is_tensor(sigma2) else sigma2)) + self.eps
        z = (r * r) / denom + 1.0
        nll = 0.5 * (self.nu + 1.0) * torch.log(z + self.eps)  # [N]
        if weight is not None:
            w = weight.clamp_min(0.0)
            return (w * nll).sum() / (w.sum() + self.eps)
        return nll.mean()


# ----------------------------
# DIP network
# ----------------------------
class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        groups = max(1, min(8, out_ch))
        self.gn = nn.GroupNorm(groups, out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x): return self.drop(self.act(self.gn(self.conv(self.pad(x)))))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.c1 = ConvGNAct(in_ch, out_ch, dropout)
        self.c2 = ConvGNAct(out_ch, out_ch, dropout)
        self.pool = nn.AvgPool2d(2)
    def forward(self, x):
        x = self.c1(x); x = self.c2(x); skip = x; x = self.pool(x); return x, skip


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c1 = ConvGNAct(in_ch + skip_ch, out_ch, dropout)
        self.c2 = ConvGNAct(out_ch, out_ch, dropout)
    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(-2) - x.size(-2); dw = skip.size(-1) - x.size(-1)
        if dh or dw: x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([x, skip], dim=1); x = self.c1(x); x = self.c2(x); return x


class DIPUNet(nn.Module):
    def __init__(self, input_depth=32, base=64, depth=5, dropout=0.0):
        super().__init__()
        chs = [base * min(2**i, 8) for i in range(depth)]
        self.downs = nn.ModuleList(); in_c = input_depth
        for out_c in chs: self.downs.append(Down(in_c, out_c, dropout)); in_c = out_c
        self.bottleneck = nn.Sequential(ConvGNAct(chs[-1], chs[-1], dropout),
                                        ConvGNAct(chs[-1], chs[-1], dropout))
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            out_c = chs[i-1] if i > 0 else base
            self.ups.append(Up(chs[i], chs[i], out_c, dropout))
        self.final = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(base, 1, kernel_size=3, padding=0))
    def forward(self, z):
        skips = []; x = z
        for d in self.downs: x, s = d(x); skips.append(s)
        x = self.bottleneck(x)
        for i, u in enumerate(self.ups): x = u(x, skips[-(i+1)])
        return self.final(x)


# ----------------------------
# NUDFT precompute & predict
# ----------------------------
@torch.no_grad()
def _grid_coords(H: int, W: int, cell_size_rad: float, device, dtype):
    l = (torch.arange(W, device=device, dtype=dtype) - (W // 2)) * cell_size_rad
    m = (torch.arange(H, device=device, dtype=dtype) - (H // 2)) * cell_size_rad
    return l, m


def precompute_uv(uu: torch.Tensor, vv: torch.Tensor, H: int, W: int, cell_size_rad: float, device, dtype):
    l, m = _grid_coords(H, W, cell_size_rad, device, dtype)
    ul = 2.0 * math.pi * (uu[:, None] * l[None, :])  # [N,W]
    vm = 2.0 * math.pi * (vv[:, None] * m[None, :])  # [N,H]
    Cul = torch.cos(ul);  Sul = torch.sin(ul)
    Cvm = torch.cos(vm);  Svm = torch.sin(vm)
    return Cul, Sul, Cvm, Svm


def predict_vis(img: torch.Tensor, Cul, Sul, Cvm, Svm) -> torch.Tensor:
    """
    img: [1,1,H,W], return [N,2] (Re, Im), exp(-iθ) convention.
    """
    I = img[0, 0]
    A = I @ Cul.t()
    B = I @ Sul.t()
    real = torch.sum(Cvm * A.t(), dim=1) - torch.sum(Svm * B.t(), dim=1)
    imag = -(torch.sum(Cvm * B.t(), dim=1) + torch.sum(Svm * A.t(), dim=1))
    return torch.stack([real, imag], dim=1)


# ----------------------------
# Gaussian helpers (init & blur)
# ----------------------------
def make_gaussian_2d(H: int, W: int, fwhm_pix: float, device, dtype, amp: float = 1.0) -> torch.Tensor:
    if fwhm_pix <= 0: fwhm_pix = max(1.0, min(H, W) / 6.0)
    sigma = fwhm_pix / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    y = (torch.arange(H, device=device, dtype=dtype) - (H // 2))[..., None]
    x = (torch.arange(W, device=device, dtype=dtype) - (W // 2))[None, ...]
    g = torch.exp(-0.5 * ((x * x + y * y) / (sigma * sigma)))
    g = g / (g.max().clamp_min(1e-12))
    return amp * g


def construct_seed_z(H: int, W: int, depth: int,
                     init_gaussian: bool, gauss_fwhm_uas: Optional[float],
                     pixel_size_uas: float, device, dtype,
                     gauss_amp: float, gauss_noise: float) -> torch.Tensor:
    if not init_gaussian:
        return torch.randn(1, depth, H, W, device=device, dtype=dtype)
    if gauss_fwhm_uas is None or gauss_fwhm_uas <= 0:
        gauss_fwhm_uas = (min(H, W) * pixel_size_uas) / 4.0
    fwhm_pix = gauss_fwhm_uas / max(pixel_size_uas, 1e-9)
    g = make_gaussian_2d(H, W, fwhm_pix, device, dtype, amp=gauss_amp).unsqueeze(0).unsqueeze(0)
    g = g.repeat(1, depth, 1, 1)
    if gauss_noise > 0: g = g + gauss_noise * torch.randn_like(g)
    return g


def gaussian_blur(img: torch.Tensor, sigma_pix: float, device, dtype) -> torch.Tensor:
    """Separable 2D Gaussian blur using reflection padding."""
    if sigma_pix is None or sigma_pix <= 0: return img
    radius = int(3.0 * sigma_pix + 0.5)
    if radius < 1: return img
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k1 = torch.exp(-0.5 * (x / sigma_pix) ** 2)
    k1 = (k1 / k1.sum()).view(1, 1, 1, -1)
    k2 = k1.transpose(2, 3)
    tmp = F.conv2d(F.pad(img, (radius, radius, 0, 0), mode="reflect"), k1)
    out = F.conv2d(F.pad(tmp, (0, 0, radius, radius), mode="reflect"), k2)
    return out


# ----------------------------
# Baseline length weights
# ----------------------------
def len_weight_from_bl(bl: torch.Tensor, scheme: str, b0: float, p: float) -> torch.Tensor:
    """
    bl: [N] in wavelengths. scheme in {'none','exp','inv'}.
      'exp' : w = exp(-(bl/b0)^p)
      'inv' : w = 1 / (1 + (bl/b0)^p)
    """
    if scheme is None or scheme == "none": return torch.ones_like(bl)
    b0 = max(float(b0), 1e-9)
    if scheme == "exp":
        w = torch.exp(- (bl / b0) ** p)
    elif scheme == "inv":
        w = 1.0 / (1.0 + (bl / b0) ** p)
    else:
        raise ValueError("Unknown length-weight scheme.")
    return w


# ----------------------------
# Config
# ----------------------------
@dataclass
class ReconConfig:
    seed: int = 42
    input_depth: int = 32
    base_ch: int = 64
    depth: int = 5
    dropout: float = 0.0
    positivity: bool = True

    # optimization
    lr: float = 1e-3
    num_iter: int = 2000
    out_every: int = 100

    # data-term weights
    amp_weight: float = 1.0
    cp_weight: float = 1.0

    # smoothness
    tv_weight: float = 0.0
    tsv_weight: float = 0.0

    # Student-t
    use_t_amp: bool = True
    t_nu_amp: float = 3.0
    t_learn_sigma_amp: bool = False
    t_init_sigma_amp: float = 1.0

    use_t_cp: bool = True
    t_nu_cp: float = 3.0
    t_learn_sigma_cp: bool = False
    t_init_sigma_cp: float = 1.0

    # flux/compactness/void
    lambda_flux: float = 0.0
    target_flux: Optional[float] = None
    lambda_compact: float = 0.0
    r_compact_uas: Optional[float] = None
    lambda_void: float = 0.0
    r_void_uas: Optional[float] = None

    # sparsity & centroid
    lambda_l1: float = 0.0
    lambda_centroid: float = 0.0

    # Gaussian init
    init_gaussian: bool = False
    gauss_fwhm_uas: Optional[float] = None
    gauss_amp: float = 1.0
    gauss_noise: float = 0.05

    # NEW: baseline-length weighting
    amp_len_scheme: str = "none"   # {'none','exp','inv'}
    amp_bl0: float = 3.0e9
    amp_len_pow: float = 2.0

    cp_len_scheme: str = "none"    # {'none','exp','inv'}
    cp_bl0: float = 5.0e9
    cp_len_pow: float = 2.0

    # NEW: learnable global image scale
    learn_global_scale: bool = False
    global_scale_init: float = 1.0

    # NEW: coarse-to-fine blur schedule (in pixels)
    blur_sigma0_pix: Optional[float] = None
    blur_sigma1_pix: float = 0.0
    blur_decay_iters: int = 20000


# ----------------------------
# Main reconstruction
# ----------------------------
def reconstruct_eht_dip(
    data_dict: Dict[str, Any],
    npix: int = 128,
    pixel_size: float = 5.0,  # μas per pixel
    device: Optional[str] = None,
    save_dir: Optional[str] = None,
    save_every: Optional[int] = None,

    # data-term weights
    amp_weight: Optional[float] = None,
    cp_weight: Optional[float] = None,

    # smoothness
    lambda_tv: Optional[float] = None,
    lambda_tsv: Optional[float] = None,

    # Student-t
    use_t_amp: Optional[bool] = None, t_nu_amp: Optional[float] = None,
    t_learn_sigma_amp: Optional[bool] = None, t_init_sigma_amp: Optional[float] = None,
    use_t_cp: Optional[bool] = None,  t_nu_cp: Optional[float] = None,
    t_learn_sigma_cp: Optional[bool] = None, t_init_sigma_cp: Optional[float] = None,

    # flux/compactness/void
    lambda_flux: Optional[float] = None, target_flux: Optional[float] = None,
    lambda_compact: Optional[float] = None, r_compact_uas: Optional[float] = None,
    lambda_void: Optional[float] = None,     r_void_uas: Optional[float] = None,

    # sparsity & centroid
    lambda_l1: Optional[float] = None,
    lambda_centroid: Optional[float] = None,

    # Gaussian init
    init_gaussian: Optional[bool] = None, gauss_fwhm_uas: Optional[float] = None,
    gauss_amp: Optional[float] = None,    gauss_noise: Optional[float] = None,

    # baseline-length weighting
    amp_len_scheme: Optional[str] = None, amp_bl0: Optional[float] = None, amp_len_pow: Optional[float] = None,
    cp_len_scheme: Optional[str] = None,  cp_bl0: Optional[float] = None,  cp_len_pow: Optional[float] = None,

    # learnable global scale
    learn_global_scale: Optional[bool] = None, global_scale_init: Optional[float] = None,

    # blur schedule
    blur_sigma0_pix: Optional[float] = None, blur_sigma1_pix: Optional[float] = None, blur_decay_iters: Optional[int] = None,

    # optimizer
    lr: Optional[float] = None, num_iter: Optional[int] = None, out_every: Optional[int] = None,

    # --- bootstrap hooks (optional) ---
    amp_extra_weights: Optional[Any] = None,
    cp_extra_weights: Optional[Any] = None,
    fixed_z: Optional[Any] = None,
    init_model_state: Optional[dict] = None,
    return_state: bool = False,
):
    dev = pick_device(device)
    dtype = default_dtype_for_device(dev)

    # config
    cfg = ReconConfig()
    # basic
    if amp_weight is not None: cfg.amp_weight = float(amp_weight)
    if cp_weight  is not None: cfg.cp_weight  = float(cp_weight)
    if lambda_tv  is not None: cfg.tv_weight  = float(lambda_tv)
    if lambda_tsv is not None: cfg.tsv_weight = float(lambda_tsv)
    if use_t_amp  is not None: cfg.use_t_amp  = bool(use_t_amp)
    if t_nu_amp   is not None: cfg.t_nu_amp   = float(t_nu_amp)
    if t_learn_sigma_amp is not None: cfg.t_learn_sigma_amp = bool(t_learn_sigma_amp)
    if t_init_sigma_amp  is not None: cfg.t_init_sigma_amp  = float(t_init_sigma_amp)
    if use_t_cp   is not None: cfg.use_t_cp   = bool(use_t_cp)
    if t_nu_cp    is not None: cfg.t_nu_cp    = float(t_nu_cp)
    if t_learn_sigma_cp is not None: cfg.t_learn_sigma_cp = bool(t_learn_sigma_cp)
    if t_init_sigma_cp  is not None: cfg.t_init_sigma_cp  = float(t_init_sigma_cp)
    if lambda_flux is not None: cfg.lambda_flux = float(lambda_flux)
    if target_flux is not None: cfg.target_flux = float(target_flux)
    if lambda_compact is not None: cfg.lambda_compact = float(lambda_compact)
    if r_compact_uas is not None: cfg.r_compact_uas = float(r_compact_uas)
    if lambda_void is not None: cfg.lambda_void = float(lambda_void)
    if r_void_uas is not None: cfg.r_void_uas = float(r_void_uas)
    if lambda_l1 is not None: cfg.lambda_l1 = float(lambda_l1)
    if lambda_centroid is not None: cfg.lambda_centroid = float(lambda_centroid)
    if init_gaussian is not None: cfg.init_gaussian = bool(init_gaussian)
    if gauss_fwhm_uas is not None: cfg.gauss_fwhm_uas = float(gauss_fwhm_uas)
    if gauss_amp is not None: cfg.gauss_amp = float(gauss_amp)
    if gauss_noise is not None: cfg.gauss_noise = float(gauss_noise)
    if amp_len_scheme is not None: cfg.amp_len_scheme = amp_len_scheme
    if amp_bl0 is not None: cfg.amp_bl0 = float(amp_bl0)
    if amp_len_pow is not None: cfg.amp_len_pow = float(amp_len_pow)
    if cp_len_scheme is not None: cfg.cp_len_scheme = cp_len_scheme
    if cp_bl0 is not None: cfg.cp_bl0 = float(cp_bl0)
    if cp_len_pow is not None: cfg.cp_len_pow = float(cp_len_pow)
    if learn_global_scale is not None: cfg.learn_global_scale = bool(learn_global_scale)
    if global_scale_init is not None: cfg.global_scale_init = float(global_scale_init)
    if blur_sigma0_pix is not None: cfg.blur_sigma0_pix = float(blur_sigma0_pix)
    if blur_sigma1_pix is not None: cfg.blur_sigma1_pix = float(blur_sigma1_pix)
    if blur_decay_iters is not None: cfg.blur_decay_iters = int(blur_decay_iters)
    if lr is not None: cfg.lr = float(lr)
    if num_iter is not None: cfg.num_iter = int(num_iter)
    if out_every is not None: cfg.out_every = int(out_every)

    H = W = int(npix)
    cell_size_uas = float(pixel_size)
    cell_size_rad = cell_size_uas * 1e-6 * (math.pi / 648000.0)

    # data
    u  = to_tensor(data_dict.get("u"),  dev, dtype);    v = to_tensor(data_dict.get("v"), dev, dtype)
    vis= to_tensor(data_dict.get("vis"),dev, dtype);    vsig = to_tensor(data_dict.get("vsigma"), dev, dtype)
    cphase  = to_tensor(np.deg2rad(data_dict.get("cphase", None)), dev, dtype)
    sigmacp = to_tensor(np.deg2rad(data_dict.get("sigmacp", None)), dev, dtype)
    u1 = to_tensor(data_dict.get("u1", None), dev, dtype); v1 = to_tensor(data_dict.get("v1", None), dev, dtype)
    u2 = to_tensor(data_dict.get("u2", None), dev, dtype); v2 = to_tensor(data_dict.get("v2", None), dev, dtype)

    if u is None or v is None or vis is None or vsig is None:
        raise ValueError("data_dict must contain 'u','v','vis','vsigma'.")

    # precompute NUDFT
    Cul, Sul, Cvm, Svm = precompute_uv(u, v, H, W, cell_size_rad, dev, dtype)

    have_cp = cphase is not None and sigmacp is not None and u1 is not None and v1 is not None and u2 is not None and v2 is not None
    if have_cp:
        u3 = -(u1 + u2); v3 = -(v1 + v2)
        Cul1, Sul1, Cvm1, Svm1 = precompute_uv(u1, v1, H, W, cell_size_rad, dev, dtype)
        Cul2, Sul2, Cvm2, Svm2 = precompute_uv(u2, v2, H, W, cell_size_rad, dev, dtype)
        Cul3, Sul3, Cvm3, Svm3 = precompute_uv(u3, v3, H, W, cell_size_rad, dev, dtype)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    net = DIPUNet(cfg.input_depth, cfg.base_ch, cfg.depth, cfg.dropout).to(dev)
    z   = construct_seed_z(H, W, cfg.input_depth, cfg.init_gaussian, cfg.gauss_fwhm_uas,
                           cell_size_uas, dev, dtype, cfg.gauss_amp, cfg.gauss_noise)

    # Warm-start hooks (optional)
    if fixed_z is not None:
        z_in = to_tensor(fixed_z, dev, dtype)
        try:
            if z_in.shape != z.shape:
                if z_in.ndim == 2: z_in = z_in[None, None, :, :]
                if z_in.ndim == 3: z_in = z_in[None, :, :, :]
        except Exception:
            pass
        if z_in is not None and tuple(z_in.shape) == tuple(z.shape):
            z = z_in.detach().clone()
    if init_model_state is not None:
        try:
            net.load_state_dict(init_model_state, strict=False)
        except Exception as e:
            print(f"[warn] init_model_state load failed: {e}")

    # optional global scale α
    if cfg.learn_global_scale:
        log_alpha = nn.Parameter(torch.log(torch.tensor(cfg.global_scale_init, device=dev, dtype=dtype)))
    else:
        log_alpha = None

    def apply_scale_and_blur(x, it: int):
        if cfg.learn_global_scale and log_alpha is not None:
            x = x * torch.exp(log_alpha)
        # blur schedule
        if cfg.blur_sigma0_pix is not None and cfg.blur_sigma0_pix > 0:
            t = min(1.0, it / max(cfg.blur_decay_iters, 1))
            sigma_now = (1.0 - t) * cfg.blur_sigma0_pix + t * cfg.blur_sigma1_pix
            if sigma_now > 0:
                x = gaussian_blur(x, sigma_now, dev, dtype)
        return x

    # losses
    amp_loss = StudentTLoss1D(cfg.t_nu_amp, cfg.t_learn_sigma_amp, cfg.t_init_sigma_amp) if cfg.use_t_amp else None
    cp_loss  = StudentTLoss1D(cfg.t_nu_cp,  cfg.t_learn_sigma_cp,  cfg.t_init_sigma_cp)  if cfg.use_t_cp  else None

    params = list(net.parameters())
    if cfg.learn_global_scale and log_alpha is not None: params += [log_alpha]
    if amp_loss is not None and amp_loss.learn_sigma:    params += [amp_loss.log_sigma]
    if cp_loss  is not None and cp_loss.learn_sigma:     params += [cp_loss.log_sigma]
    opt = torch.optim.Adam(params, lr=cfg.lr)

    # baseline-length weights
    bl = torch.sqrt(u*u + v*v)
    w_amp = len_weight_from_bl(bl, cfg.amp_len_scheme, cfg.amp_bl0, cfg.amp_len_pow)
    if have_cp:
        l1 = torch.sqrt(u1*u1 + v1*v1)
        l2 = torch.sqrt(u2*u2 + v2*v2)
        l3 = torch.sqrt((u1+u2)**2 + (v1+v2)**2)
        bmax = torch.maximum(l1, torch.maximum(l2, l3))
        w_cp = len_weight_from_bl(bmax, cfg.cp_len_scheme, cfg.cp_bl0, cfg.cp_len_pow)
    else:
        w_cp = None

    # Apply extra per-sample weights if provided (e.g., bootstrap multipliers)
    if amp_extra_weights is not None:
        w_extra = to_tensor(amp_extra_weights, dev, dtype)
        if w_extra is not None:
            w_amp = (w_amp * w_extra)
    if have_cp and (cp_extra_weights is not None):
        w_extra_cp = to_tensor(cp_extra_weights, dev, dtype)
        if w_extra_cp is not None:
            w_cp = w_extra_cp if w_cp is None else (w_cp * w_extra_cp)

    # saving
    if save_dir: os.makedirs(save_dir, exist_ok=True)

    # ----------- initialize a valid best_img before the loop -----------
    with torch.no_grad():
        img0 = net(z)
        img0 = F.softplus(img0) if cfg.positivity else img0
        img0 = apply_scale_and_blur(img0, it=0)
        best_img = img0.detach().clone()
    best_obj = float("inf")

    # If no iterations requested, return the initialized image/state
    if cfg.num_iter is None or cfg.num_iter <= 0:
        im = best_img[0, 0].detach().cpu().numpy()
        if return_state:
            state = {"z": z.detach().cpu().numpy(), "model_state_dict": net.state_dict()}
            return {"image": im, "state": state}
        return im
    # -----------------------------------------------------------------------

    for it in range(1, cfg.num_iter + 1):
        opt.zero_grad(set_to_none=True)

        img = net(z)                      # [1,1,H,W]
        img = F.softplus(img) if cfg.positivity else img
        img_s = apply_scale_and_blur(img, it)

        # vis amps
        pred = predict_vis(img_s, Cul, Sul, Cvm, Svm)  # [N,2]
        amp = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)
        r = (amp - torch.abs(vis)).flatten()
        if amp_loss is not None:
            d_amp = amp_loss(r, sigma=torch.abs(vsig), weight=w_amp)
        else:
            rnorm = r / (torch.abs(vsig) + 1e-12)
            d_amp = (w_amp * rnorm * rnorm).sum() / (w_amp.sum() + 1e-12)

        # closure phase
        if have_cp:
            V1 = predict_vis(img_s, Cul1, Sul1, Cvm1, Svm1)
            V2 = predict_vis(img_s, Cul2, Sul2, Cvm2, Svm2)
            V3 = predict_vis(img_s, Cul3, Sul3, Cvm3, Svm3)
            ph1 = torch.atan2(V1[:, 1], V1[:, 0])
            ph2 = torch.atan2(V2[:, 1], V2[:, 0])
            ph3 = torch.atan2(V3[:, 1], V3[:, 0])
            cp_pred = wrap_phase(ph1 + ph2 + ph3)
            dphi = wrap_phase_diff(cp_pred, cphase)
            if cp_loss is not None:
                d_cp = cp_loss(dphi, sigma=sigmacp, weight=w_cp)
            else:
                rcp = dphi / (sigmacp + 1e-12)
                if w_cp is not None:
                    d_cp = (w_cp * rcp * rcp).sum() / (w_cp.sum() + 1e-12)
                else:
                    d_cp = (rcp * rcp).mean()
        else:
            d_cp = torch.tensor(0.0, device=dev, dtype=dtype)

        # smoothness
        d_tv  = tv_loss(img_s)
        d_tsv = tsv_loss(img_s)

        # sparsity
        d_l1 = l1_loss_mean(img_s)

        # flux/compactness/void
        im2d = img_s[0, 0]
        if cfg.lambda_flux > 0.0 and cfg.target_flux is not None:
            total_flux = torch.sum(im2d)
            scale = max(cfg.target_flux, 1.0)
            d_flux = ((total_flux - cfg.target_flux) / scale) ** 2
        else:
            d_flux = torch.tensor(0.0, device=dev, dtype=dtype)

        if cfg.lambda_compact > 0.0 or cfg.lambda_void > 0.0:
            mask_out, mask_in = radial_masks(H, W, cell_size_uas, cfg.r_compact_uas, cfg.r_void_uas, dev, dtype)
            d_compact = torch.mean(im2d * mask_out) if cfg.lambda_compact > 0.0 else torch.tensor(0.0, device=dev, dtype=dtype)
            d_void    = torch.mean(im2d * mask_in)  if cfg.lambda_void     > 0.0 else torch.tensor(0.0, device=dev, dtype=dtype)
        else:
            d_compact = torch.tensor(0.0, device=dev, dtype=dtype)
            d_void    = torch.tensor(0.0, device=dev, dtype=dtype)

        d_cent = centroid_penalty(img_s, dev, dtype)

        # objective
        obj = (cfg.amp_weight * d_amp +
               cfg.cp_weight  * d_cp  +
               cfg.tv_weight  * d_tv  +
               cfg.tsv_weight * d_tsv +
               cfg.lambda_l1  * d_l1  +
               cfg.lambda_flux     * d_flux +
               cfg.lambda_compact  * d_compact +
               cfg.lambda_void     * d_void +
               cfg.lambda_centroid * d_cent)

        obj.backward()
        opt.step()

        # logging
        if (it % (cfg.out_every if out_every is None else out_every) == 0) or it == 1:
            with torch.no_grad():
                w_amp_log = w_amp.detach().clamp_min(0.0)
                rnorm_amp_log = r.detach() / (torch.abs(vsig).flatten() + 1e-12)
                chi2_amp = (w_amp_log * rnorm_amp_log * rnorm_amp_log).sum() / (w_amp_log.sum() + 1e-12)

                if have_cp:
                    rcp_log = dphi.detach() / (torch.abs(sigmacp).flatten() + 1e-12)
                    if w_cp is not None:
                        w_cp_log = w_cp.detach().clamp_min(0.0)
                        chi2_cp = (w_cp_log * rcp_log * rcp_log).sum() / (w_cp_log.sum() + 1e-12)
                    else:
                        chi2_cp = (rcp_log * rcp_log).mean()
                else:
                    chi2_cp = torch.tensor(0.0, device=dev, dtype=dtype)

            def _format_regularizer(name: str, value: torch.Tensor, weight: float) -> str:
                value_text = f"{name}={_tensor_item(value):.4g}"
                if float(weight) != 0.0:
                    value_text = _color_text(value_text, "magenta")
                weight_text = _color_text(f"λ={float(weight):.4g}", "green")
                return f"{value_text}({weight_text})"

            msg_parts = [
                _color_text(f"chi2_amp={_tensor_item(chi2_amp):.4g}", "yellow"),
                _color_text(f"chi2_cp={_tensor_item(chi2_cp):.4g}", "cyan"),
                _format_regularizer("tv", d_tv, cfg.tv_weight),
                _format_regularizer("tsv", d_tsv, cfg.tsv_weight),
                _format_regularizer("l1", d_l1, cfg.lambda_l1),
                _format_regularizer("flux", d_flux, cfg.lambda_flux),
                _format_regularizer("comp", d_compact, cfg.lambda_compact),
                _format_regularizer("void", d_void, cfg.lambda_void),
                _format_regularizer("cent", d_cent, cfg.lambda_centroid),
            ]
            msg = f"[{it:5d}/{cfg.num_iter}] " + " ".join(msg_parts)
            if cfg.learn_global_scale and log_alpha is not None:
                msg += f" alpha={torch.exp(log_alpha).detach().cpu().item():.3g}"
            if amp_loss is not None and amp_loss.learn_sigma:
                msg += f" s_amp={torch.exp(amp_loss.log_sigma).detach().cpu().item():.3g}"
            if have_cp and cp_loss is not None and cp_loss.learn_sigma:
                msg += f" s_cp={torch.exp(cp_loss.log_sigma).detach().cpu().item():.3g}"
            print(msg)

        # track best by data terms only; guard against NaNs
        with torch.no_grad():
            data_val = (cfg.amp_weight * d_amp + cfg.cp_weight * d_cp).detach()
            if torch.isfinite(data_val):
                if data_val.cpu().item() < best_obj:
                    best_obj = data_val.cpu().item()
                    best_img = img_s.detach().clone()

        # optional saves
        if save_dir and save_every and (it % save_every == 0):
            np.save(os.path.join(save_dir, f"img_{it:05d}.npy"), img_s[0, 0].detach().cpu().numpy())

    im = best_img[0, 0].detach().cpu().numpy()
    if return_state:
        state = {
            "z": z.detach().cpu().numpy(),
            "model_state_dict": net.state_dict()
        }
        return {"image": im, "state": state}
    return im



# =============================================================================
# Multiplier (weighted) bootstrap 
# =============================================================================

# --- helpers for bootstrap methods ---

def _dirichlet_weights(N: int, rng: np.random.Generator, weight_floor: float = 0.0) -> np.ndarray:
    """
    Bayesian bootstrap weights ~ Dirichlet(1,...,1).
    Optionally shrink away from 0 by 'weight_floor' (in probability mass).
    Returns shape (N,) float32.
    """
    if N <= 0:
        return np.zeros((0,), dtype=np.float32)
    # Gamma(1,1) ~ Exponential(1). Normalize to Dirichlet(1,...,1).
    g = rng.gamma(shape=1.0, scale=1.0, size=N).astype(np.float64)
    g /= g.sum() if g.sum() > 0 else 1.0
    if weight_floor > 0.0:
        eps = float(weight_floor)
        g = (1.0 - eps) * g + eps / N
    return g.astype(np.float32)


def _pairs_weights(N: int, rng: np.random.Generator, weight_floor: float = 0.0) -> np.ndarray:
    """
    Pairs bootstrap via resampling indices with replacement.
    Returns counts normalized to sum 1 (optionally shrunk away from zeros by weight_floor).
    """
    if N <= 0:
        return np.zeros((0,), dtype=np.float32)
    idx = rng.integers(0, N, size=N, endpoint=False)
    counts = np.bincount(idx, minlength=N).astype(np.float64)
    w = counts / counts.sum() if counts.sum() > 0 else counts
    if weight_floor > 0.0:
        eps = float(weight_floor)
        w = (1.0 - eps) * w + eps / N
    return w.astype(np.float32)


def _draw_bootstrap_weights(n: int, kind: str, rng: np.random.Generator) -> Optional[np.ndarray]:
    """
    Draw nonnegative weights for bootstrap over n items.

    kind:
      - 'poisson'   : counts ~ Poisson(1), standard (Bayesian) bootstrap
      - 'dirichlet' : w ~ Dirichlet(1,...,1) scaled to sum to n

    Returns a float32 array of shape [n]. Ensures not all zero (falls back to ones).
    """
    if n is None or n <= 0:
        return None
    if kind.lower() == "poisson":
        w = rng.poisson(1.0, size=n).astype(np.float32)
    elif kind.lower() == "dirichlet":
        w = rng.dirichlet(alpha=np.ones(n)).astype(np.float32) * float(n)
    else:
        raise ValueError(f"Unknown bootstrap kind: {kind}. Use 'poisson' or 'dirichlet'.")
    if np.sum(w) <= 0:
        w = np.ones(n, dtype=np.float32)
    return w



def reconstruct_eht_dip_bootstrap(
    data_dict: Dict[str, Any],
    B: int = 200,

    # Choose bootstrap kind
    bootstrap_kind: str = "poisson",  # {'poisson','dirichlet'}

    # Deprecated args (kept for backward compatibility; ignored)
    multiplier_kind_amp: Optional[str] = None,
    multiplier_kind_cp: Optional[str] = None,

    # Warm-start & schedule
    seed: int = 12345,
    do_baseline_map: bool = True,
    warm_start: bool = True,
    short_num_iter: Optional[int] = 600,   # per-replicate iters; set None to keep recon_kwargs['num_iter']
    reuse_z: bool = True,

    # Saving / memory
    save_dir: Optional[str] = None,
    save_stack: bool = False,

    # Forwarded to reconstruct_eht_dip (priors, t params, etc.)
    **recon_kwargs
) -> Dict[str, Any]:
    """
    Bootstrap imaging for EHT DIP via nonparametric weights on the data terms.

    - Optional baseline run to warm-start (state and input z).
    - For each replicate, draw bootstrap weights for amplitude data and (if present) closure phases,
      then run a short reconstruction with those weights applied.
    """
    # Back-compat notice (only once)
    if multiplier_kind_amp is not None or multiplier_kind_cp is not None:
        print("[info] reconstruct_eht_dip_bootstrap: 'multiplier_kind_*' args are deprecated "
              "and ignored. Using bootstrap_kind='%s'." % bootstrap_kind)

    rng = np.random.default_rng(seed)

    # 1) Optional baseline MAP (for warm-start + reference)
    baseline_state = None
    baseline_image = None
    if do_baseline_map or warm_start or reuse_z:
        kw0 = dict(recon_kwargs)         # avoid duplicate kwargs
        kw0["return_state"] = True
        base_out = reconstruct_eht_dip(
            data_dict=data_dict,
            **kw0
        )
        baseline_image = base_out["image"]
        baseline_state = base_out["state"]

    # Shared warm-start hooks
    fixed_z = baseline_state["z"] if (reuse_z and baseline_state is not None) else None
    init_model_state = baseline_state["model_state_dict"] if (warm_start and baseline_state is not None) else None

    # Determine output shape
    if baseline_image is not None:
        H, W = baseline_image.shape
    else:
        npix = int(recon_kwargs.get("npix", 128))
        H, W = npix, npix

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    imgs = np.zeros((B, H, W), dtype=np.float32)

    # Counts
    N_amp = int(np.asarray(data_dict["u"]).size)
    have_cp = all(k in data_dict for k in ("cphase", "sigmacp", "u1", "v1", "u2", "v2"))
    N_cp = int(np.asarray(data_dict["cphase"]).size) if have_cp else 0

    for b in range(B):
        # 2) Draw bootstrap weights
        w_amp = _draw_bootstrap_weights(N_amp, bootstrap_kind, rng)
        w_cp  = _draw_bootstrap_weights(N_cp,  bootstrap_kind, rng) if have_cp else None

        # 3) Prepare kwargs, avoiding duplicate keys
        kw = dict(recon_kwargs)
        if short_num_iter is not None:
            kw["num_iter"] = int(short_num_iter)
        kw["return_state"] = False

        out_img = reconstruct_eht_dip(
            data_dict=data_dict,
            amp_extra_weights=w_amp,
            cp_extra_weights=w_cp,
            fixed_z=fixed_z,
            init_model_state=init_model_state,
            **kw
        )
        imgs[b] = np.asarray(out_img, dtype=np.float32)

        if save_dir is not None:
            np.save(os.path.join(save_dir, f"bootstrap_img_{b:04d}.npy"), imgs[b])

    # 4) Aggregate
    mean = imgs.mean(axis=0)
    std  = imgs.std(axis=0, ddof=1) if B > 1 else np.zeros_like(mean)
    q16  = np.quantile(imgs, 0.16, axis=0)
    q84  = np.quantile(imgs, 0.84, axis=0)

    result = {
        "baseline_image": baseline_image,
        "mean": mean,
        "std": std,
        "q16": q16,
        "q84": q84,
    }

    if save_stack:
        stack_path = os.path.join(save_dir or ".", "bootstrap_stack.npy")
        np.save(stack_path, imgs)
        result["stack_path"] = stack_path
    else:
        result["images"] = imgs

    return result



# =============================================================================
# Minimal self-test (optional)
# =============================================================================

if __name__ == "__main__":
    # This block is a quick smoke test with synthetic data if desired. This is just placed here for debugging purposes...
    print("eht_dip_enhanced3.py loaded. (Run your own driver to feed uv data.)")
