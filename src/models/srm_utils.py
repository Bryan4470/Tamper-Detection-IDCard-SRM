"""
SRM (Spatial Rich Model) Filter Utilities

30 fixed high-pass filter kernels that extract noise-level residuals from images.
These kernels capture resampling artifacts, JPEG double-compression boundaries,
and sharpening traces that persist even when colour is normalised.

Reference: Fridrich & Kodovsky, "Rich Models for Steganalysis of Digital Images", 2012
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_srm_kernels() -> np.ndarray:
    """
    Build 30 SRM high-pass filter kernels of size 5x5.

    Returns:
        Array of shape (30, 5, 5), normalized so sum-of-abs = 1.
    """
    kernels = []

    # ---- Order-1 finite differences (5 filters) ----
    # Horizontal right, left; Vertical down, up; Diagonal NW-SE
    diffs_1 = [
        ([2, 2], [2, 3]),        # center - right
        ([2, 2], [2, 1]),        # center - left
        ([2, 2], [3, 2]),        # center - down
        ([2, 2], [1, 2]),        # center - up
        ([2, 2], [3, 3]),        # center - diag NW-SE
    ]
    for (r0, c0), (r1, c1) in diffs_1:
        k = np.zeros((5, 5), dtype=np.float32)
        k[r0, c0] = 1.0
        k[r1, c1] = -1.0
        kernels.append(k)

    # ---- Order-2 finite differences (10 filters) ----
    # Horizontal, Vertical, two Diagonals, 2D Laplacian, and 5 extended variants
    order2_specs = [
        # (center_weight, list of (row, col) for -1 entries)
        (2,  [(2, 1), (2, 3)]),           # horizontal Laplacian
        (2,  [(1, 2), (3, 2)]),           # vertical Laplacian
        (2,  [(1, 1), (3, 3)]),           # diagonal Laplacian NW-SE
        (2,  [(1, 3), (3, 1)]),           # diagonal Laplacian NE-SW
        (4,  [(1, 2), (2, 1), (2, 3), (3, 2)]),   # 2D Laplacian (4-neighbour)
        (4,  [(2, 0), (2, 1), (2, 3), (2, 4)]),   # horizontal extended
        (4,  [(0, 2), (1, 2), (3, 2), (4, 2)]),   # vertical extended
        (4,  [(0, 0), (1, 1), (3, 3), (4, 4)]),   # diagonal extended NW-SE
        (4,  [(0, 4), (1, 3), (3, 1), (4, 0)]),   # diagonal extended NE-SW
        (8,  [(1, 1), (1, 2), (1, 3),
              (2, 1),          (2, 3),
              (3, 1), (3, 2), (3, 3)]),   # 3x3 Laplacian
    ]
    for center_w, neg_coords in order2_specs:
        k = np.zeros((5, 5), dtype=np.float32)
        k[2, 2] = float(center_w)
        for r, c in neg_coords:
            k[r, c] = -1.0
        kernels.append(k)

    # ---- Order-3 finite differences (10 filters) ----
    order3_rows = [
        # Format: list of (row, col, weight)
        [(2, 1, -1), (2, 2, 3), (2, 3, -3), (2, 4, 1)],          # H right
        [(2, 0, 1),  (2, 1, -3), (2, 2, 3), (2, 3, -1)],          # H left
        [(1, 2, -1), (2, 2, 3), (3, 2, -3), (4, 2, 1)],           # V down
        [(0, 2, 1),  (1, 2, -3), (2, 2, 3), (3, 2, -1)],          # V up
        [(1, 1, -1), (2, 2, 3), (3, 3, -3)],                      # Diag NW-SE (3-tap)
        [(1, 3, -1), (2, 2, 3), (3, 1, -3)],                      # Diag NE-SW (3-tap)
        [(2, 0, 1),  (2, 1, -4), (2, 2, 6), (2, 3, -4), (2, 4, 1)],  # H 4th-order
        [(0, 2, 1),  (1, 2, -4), (2, 2, 6), (3, 2, -4), (4, 2, 1)],  # V 4th-order
        [(1, 2, -1), (2, 1, -1), (2, 2, 4), (2, 3, -1), (3, 2, -1)], # Cross 3rd
        [(0, 0, 1),  (1, 1, -3), (2, 2, 3), (3, 3, -1)],              # Diag NW-SE 4-tap
    ]
    for spec in order3_rows:
        k = np.zeros((5, 5), dtype=np.float32)
        for r, c, w in spec:
            k[r, c] = float(w)
        kernels.append(k)

    # ---- Averaging / prediction residuals (5 filters) ----
    # Square-5 centre prediction (nearest 8 neighbours)
    k = np.zeros((5, 5), dtype=np.float32)
    k[1:4, 1:4] = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    kernels.append(k)

    # Square-5 full (all 24 neighbours predict centre)
    k = -np.ones((5, 5), dtype=np.float32)
    k[2, 2] = 24.0
    kernels.append(k)

    # Ring filter (border vs centre)
    k = np.zeros((5, 5), dtype=np.float32)
    k[0, :] = -1; k[4, :] = -1; k[:, 0] = -1; k[:, 4] = -1
    k[2, 2] = 16.0
    kernels.append(k)

    # Checkerboard residual
    k = np.zeros((5, 5), dtype=np.float32)
    for i in range(5):
        for j in range(5):
            if i == 2 and j == 2:
                k[i, j] = 1.0
            elif (i + j) % 2 == 0:
                k[i, j] = -1.0 / 12.0
    kernels.append(k)

    # Cross-shaped (horizontal + vertical bars)
    k = np.zeros((5, 5), dtype=np.float32)
    k[2, :] = -1.0; k[:, 2] = -1.0; k[2, 2] = 8.0
    kernels.append(k)

    assert len(kernels) == 30, f"Expected 30 kernels, got {len(kernels)}"

    # Normalise each kernel: divide by sum of absolute values (SRM paper convention)
    normalized = []
    for k in kernels:
        s = np.abs(k).sum()
        normalized.append(k / s if s > 0 else k)

    return np.stack(normalized, axis=0)  # (30, 5, 5)


class SRMFilter(nn.Module):
    """
    Applies 30 fixed SRM high-pass filters to an RGB image.

    Input:  (B, 3, H, W)  — normalised RGB tensor
    Output: (B, 30, H, W) — noise residual maps, clamped to [-2, 2]
    """

    TRUNCATION = 2.0  # T in the original SRM paper

    def __init__(self):
        super().__init__()
        kernels_np = _build_srm_kernels()  # (30, 5, 5)

        # Each output channel averages across the 3 RGB input channels.
        # weights shape: (out=30, in=3, 5, 5)
        weights = np.zeros((30, 3, 5, 5), dtype=np.float32)
        for k in range(30):
            for c in range(3):
                weights[k, c] = kernels_np[k] / 3.0

        weight_tensor = torch.from_numpy(weights)
        self.register_buffer('weight', weight_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised RGB in [0, 1] or ImageNet-normalised
        Returns:
            (B, 30, H, W) noise residual maps
        """
        out = F.conv2d(x, self.weight, bias=None, stride=1, padding=2)
        return out.clamp(-self.TRUNCATION, self.TRUNCATION)
