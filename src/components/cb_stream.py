"""
CB (Chrominance-Blue) Stream Components for Three-Stream Tamper Detection.

Extracts Cb channel from RGB images, divides key ID card regions into patches,
encodes them, and aggregates with attention pooling.

Adapted from face-tamper-inference/src/models/cb_utils.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from typing import Dict, Optional


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


def extract_cb_channel(rgb: torch.Tensor, input_range: str = 'normalized') -> torch.Tensor:
    """
    Extract Cb channel from RGB images.

    Args:
        rgb: RGB tensor (B, 3, H, W)
        input_range: 'normalized' for [-1,1] input (from transforms.Normalize),
                     'zero_one' for [0,1] input

    Returns:
        Cb channel tensor (B, 1, H, W)
    """
    if input_range == 'normalized':
        # Convert from [-1, 1] to [0, 1]
        rgb = (rgb + 1.0) / 2.0

    ycbcr = rgb_to_ycbcr(rgb)
    return ycbcr[:, 1:2, :, :]


class BackgroundRegionExtractor(nn.Module):
    """Extracts Cb patches from key regions of ID card images."""

    def __init__(self, regions_config: str, patch_grid_size: int = 4, fixed_region_size: int = 64):
        """
        Args:
            regions_config: Path to YAML file defining regions
            patch_grid_size: Grid size for patches (e.g., 4 = 4x4 = 16 patches per region)
            fixed_region_size: Fixed size to resize each region before patching (scaled up for 256x256)
        """
        super().__init__()
        with open(regions_config, 'r') as f:
            self.regions = yaml.safe_load(f)
        self.patch_grid_size = patch_grid_size
        self.fixed_region_size = fixed_region_size

    def forward(self, cb_channel: torch.Tensor) -> torch.Tensor:
        """
        Extract Cb patches from defined regions.

        Args:
            cb_channel: Cb tensor (B, 1, H, W)

        Returns:
            Patches tensor (B, N_total, 1, P_h, P_w)
            N_total = num_regions * patch_grid_size^2
        """
        B, C, H, W = cb_channel.shape
        all_patches = []

        for region_info in self.regions.values():
            x1 = int(region_info['x1'] * W)
            y1 = int(region_info['y1'] * H)
            x2 = int(region_info['x2'] * W)
            y2 = int(region_info['y2'] * H)

            # Clamp coordinates
            x1, x2 = max(0, x1), min(W, x2)
            y1, y2 = max(0, y1), min(H, y2)

            # Extract region
            region = cb_channel[:, :, y1:y2, x1:x2]
            patches = self._divide_into_patches(region)
            all_patches.append(patches)

        return torch.cat(all_patches, dim=1)

    def _divide_into_patches(self, region: torch.Tensor) -> torch.Tensor:
        """Divide region into non-overlapping patches."""
        B, C, h, w = region.shape

        # Resize to fixed size
        region = F.interpolate(
            region,
            size=(self.fixed_region_size, self.fixed_region_size),
            mode='bilinear',
            align_corners=False
        )

        patch_h = self.fixed_region_size // self.patch_grid_size
        patch_w = self.fixed_region_size // self.patch_grid_size

        # Reshape into patches
        region = region.reshape(
            B, C,
            self.patch_grid_size, patch_h,
            self.patch_grid_size, patch_w
        )
        region = region.permute(0, 2, 4, 1, 3, 5)  # B, grid_h, grid_w, C, patch_h, patch_w

        K = self.patch_grid_size ** 2
        return region.reshape(B, K, C, patch_h, patch_w)

    def get_num_patches(self) -> int:
        return len(self.regions) * (self.patch_grid_size ** 2)


class CbEncoder(nn.Module):
    """
    Lightweight CNN encoder for Cb patches.
    Encodes each patch from (1, 16, 16) to a 128-dim feature vector.
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            # 1 x 16 x 16 -> 32 x 8 x 8
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32 x 8 x 8 -> 64 x 4 x 4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 x 4 x 4 -> 128 x 2 x 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Global average pooling -> 128 x 1 x 1
            nn.AdaptiveAvgPool2d(1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Encode patches to feature vectors.

        Args:
            patches: (B, N, 1, P_h, P_w) tensor of Cb patches

        Returns:
            features: (B, N, feature_dim) tensor of patch features
        """
        B, N, C, H, W = patches.shape

        # Flatten batch and patches dimensions
        patches = patches.reshape(B * N, C, H, W)

        # Encode
        features = self.encoder(patches)  # (B*N, 128, 1, 1)
        features = features.view(B, N, self.feature_dim)

        return features


class CbAggregator(nn.Module):
    """
    Attention-weighted aggregation of patch features.
    Produces a single feature vector per image.
    """

    def __init__(self, input_dim: int = 128, output_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate patch features with attention.

        Args:
            patch_features: (B, N, input_dim) tensor of patch features

        Returns:
            aggregated: (B, output_dim) tensor
        """
        B, N, D = patch_features.shape

        # Compute attention weights
        attn_scores = self.attention(patch_features)  # (B, N, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, N, 1)

        # Weighted sum
        aggregated = (patch_features * attn_weights).sum(dim=1)  # (B, input_dim)

        # Project to output dimension
        aggregated = self.projection(aggregated)  # (B, output_dim)

        return aggregated

    def forward_with_features(self, patch_features: torch.Tensor) -> tuple:
        """
        Aggregate patch features with attention and return intermediate features.

        Args:
            patch_features: (B, N, input_dim) tensor of patch features

        Returns:
            aggregated: (B, output_dim) tensor
            patch_features: (B, N, input_dim) original patch features for loss
        """
        aggregated = self.forward(patch_features)
        return aggregated, patch_features


class CbStream(nn.Module):
    """
    Complete CB Stream module combining extraction, encoding, and aggregation.

    Pipeline:
    1. Extract Cb channel from RGB
    2. Extract patches from 5 ID card regions
    3. Encode each patch with lightweight CNN
    4. Aggregate with attention pooling
    """

    def __init__(
        self,
        regions_config: str,
        patch_grid_size: int = 4,
        fixed_region_size: int = 64,
        encoder_dim: int = 128,
        output_dim: int = 256,
    ):
        """
        Args:
            regions_config: Path to YAML file defining ID card regions
            patch_grid_size: Grid size for patches (4 = 4x4 = 16 patches per region)
            fixed_region_size: Fixed size for each region (64 for 256x256 input)
            encoder_dim: Output dimension of patch encoder
            output_dim: Final output dimension (256 for fusion)
        """
        super().__init__()

        self.region_extractor = BackgroundRegionExtractor(
            regions_config=regions_config,
            patch_grid_size=patch_grid_size,
            fixed_region_size=fixed_region_size,
        )

        self.encoder = CbEncoder(
            in_channels=1,
            feature_dim=encoder_dim,
        )

        self.aggregator = CbAggregator(
            input_dim=encoder_dim,
            output_dim=output_dim,
        )

        self.num_patches = self.region_extractor.get_num_patches()

    def forward(self, rgb: torch.Tensor, return_patch_features: bool = False):
        """
        Process RGB image through CB stream.

        Args:
            rgb: RGB tensor (B, 3, H, W) normalized to [-1, 1]
            return_patch_features: If True, also return patch features for CB loss

        Returns:
            cb_output: (B, output_dim) aggregated CB features
            cb_patch_features: (B, N, encoder_dim) if return_patch_features=True
        """
        # Extract Cb channel
        cb_channel = extract_cb_channel(rgb, input_range='normalized')

        # Extract patches from regions
        patches = self.region_extractor(cb_channel)

        # Encode patches
        patch_features = self.encoder(patches)

        # Aggregate with attention
        cb_output = self.aggregator(patch_features)

        if return_patch_features:
            return cb_output, patch_features

        return cb_output


if __name__ == '__main__':
    # Test CB stream
    import os
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/regions.yaml')

    # Create dummy regions config for testing
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    if not os.path.exists(config_path):
        test_regions = {
            'face': {'x1': 0.62, 'y1': 0.21, 'x2': 0.98, 'y2': 0.88},
            'id_number': {'x1': 0.02, 'y1': 0.21, 'x2': 0.38, 'y2': 0.32},
        }
        with open(config_path, 'w') as f:
            yaml.dump(test_regions, f)

    cb_stream = CbStream(regions_config=config_path)

    # Test with dummy input (normalized to [-1, 1])
    dummy_input = torch.randn(2, 3, 256, 256)

    cb_out, patch_feats = cb_stream(dummy_input, return_patch_features=True)
    print(f"CB output shape: {cb_out.shape}")  # Expected: (2, 256)
    print(f"Patch features shape: {patch_feats.shape}")  # Expected: (2, 32, 128) for 2 regions
