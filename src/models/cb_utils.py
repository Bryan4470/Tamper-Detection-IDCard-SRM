"""
Cb Channel Utilities for Tamper Detection

Provides:
1. RGB to YCbCr color space conversion
2. Background region patch extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from typing import Dict


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to YCbCr color space (ITU-R BT.601).

    Args:
        rgb: RGB tensor (B, 3, H, W) with values in [0, 1]

    Returns:
        YCbCr tensor (B, 3, H, W)
    """
    rgb = rgb.float()
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.5 * b
    cr = 0.5 * r - 0.419 * g - 0.081 * b

    return torch.cat([y, cb, cr], dim=1)


def extract_cb_channel(rgb: torch.Tensor) -> torch.Tensor:
    """Extract Cb channel from RGB images."""
    ycbcr = rgb_to_ycbcr(rgb)
    return ycbcr[:, 1:2, :, :]


class BackgroundRegionExtractor(nn.Module):
    """Extracts Cb patches from key regions of ID card images."""

    def __init__(self, regions_config: str, patch_grid_size: int = 4):
        """
        Args:
            regions_config: Path to YAML file defining regions
            patch_grid_size: Grid size for patches (e.g., 4 = 4x4 = 16 patches per region)
        """
        super().__init__()
        with open(regions_config, 'r') as f:
            self.regions = yaml.safe_load(f)
        self.patch_grid_size = patch_grid_size

    def forward(self, cb_channel: torch.Tensor) -> torch.Tensor:
        """
        Extract Cb patches from defined regions.

        Args:
            cb_channel: Cb tensor (B, 1, H, W)

        Returns:
            Patches tensor (B, N_total, 1, P_h, P_w)
        """
        B, C, H, W = cb_channel.shape
        all_patches = []

        for region_info in self.regions.values():
            x1, y1 = int(region_info['x1'] * W), int(region_info['y1'] * H)
            x2, y2 = int(region_info['x2'] * W), int(region_info['y2'] * H)

            region = cb_channel[:, :, y1:y2, x1:x2]
            patches = self._divide_into_patches(region)
            all_patches.append(patches)

        return torch.cat(all_patches, dim=1)

    def _divide_into_patches(self, region: torch.Tensor) -> torch.Tensor:
        """Divide region into non-overlapping patches."""
        B, C, h, w = region.shape
        fixed_size = 56

        region = F.interpolate(region, size=(fixed_size, fixed_size),
                               mode='bilinear', align_corners=False)

        patch_h = fixed_size // self.patch_grid_size
        patch_w = fixed_size // self.patch_grid_size

        region = region.reshape(B, C, self.patch_grid_size, patch_h,
                                self.patch_grid_size, patch_w)
        region = region.permute(0, 2, 4, 1, 3, 5)

        K = self.patch_grid_size ** 2
        return region.reshape(B, K, C, patch_h, patch_w)

    def get_num_patches(self) -> int:
        return len(self.regions) * (self.patch_grid_size ** 2)
