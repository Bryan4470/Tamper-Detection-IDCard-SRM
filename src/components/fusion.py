"""
Three-Stream Fusion Module for RGB + SRM + CB Integration.

Fuses the RGB-SRM features (2048-dim spatial) with CB features (256-dim vector)
to produce final features for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ChannelAttention


class ThreeStreamFusionModule(nn.Module):
    """
    Fuses RGB-SRM spatial features with CB global features.

    Architecture:
    1. RGB-SRM features: (B, 2048, H, W) from FeatureFusionModule
    2. CB features: (B, 256) from CbAggregator
    3. Fusion: Expand CB to spatial, concatenate, reduce with attention
    """

    def __init__(
        self,
        rgb_srm_dim: int = 2048,
        cb_dim: int = 256,
        output_dim: int = 2048,
        spatial_size: int = 8,
    ):
        """
        Args:
            rgb_srm_dim: Channel dimension of RGB-SRM features
            cb_dim: Dimension of CB features
            output_dim: Output channel dimension
            spatial_size: Spatial size of RGB-SRM features (8x8 for 256x256 input)
        """
        super().__init__()
        self.rgb_srm_dim = rgb_srm_dim
        self.cb_dim = cb_dim
        self.output_dim = output_dim
        self.spatial_size = spatial_size

        # Project CB features to match spatial feature dimension
        self.cb_projection = nn.Sequential(
            nn.Linear(cb_dim, rgb_srm_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_srm_dim // 4, rgb_srm_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Fusion convolution: RGB-SRM + projected CB
        fusion_input_dim = rgb_srm_dim + rgb_srm_dim // 2
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_input_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

        # Channel attention for refinement
        self.channel_attention = ChannelAttention(output_dim, ratio=16)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb_srm_features: torch.Tensor, cb_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse RGB-SRM and CB features.

        Args:
            rgb_srm_features: (B, 2048, H, W) spatial features from RGB-SRM fusion
            cb_features: (B, 256) global CB features

        Returns:
            fused_features: (B, 2048, H, W) fused spatial features
        """
        B, C, H, W = rgb_srm_features.shape

        # Project CB features
        cb_projected = self.cb_projection(cb_features)  # (B, 1024)

        # Expand CB to spatial dimensions
        cb_spatial = cb_projected.unsqueeze(-1).unsqueeze(-1)  # (B, 1024, 1, 1)
        cb_spatial = cb_spatial.expand(-1, -1, H, W)  # (B, 1024, H, W)

        # Concatenate along channel dimension
        combined = torch.cat([rgb_srm_features, cb_spatial], dim=1)  # (B, 3072, H, W)

        # Fuse with convolution
        fused = self.fusion_conv(combined)  # (B, 2048, H, W)

        # Apply channel attention
        fused = fused + fused * self.channel_attention(fused)

        return fused


class ChannelAttentionFusion(nn.Module):
    """
    Alternative fusion using channel-wise attention gating.
    CB features modulate which RGB-SRM channels are important.
    """

    def __init__(
        self,
        rgb_srm_dim: int = 2048,
        cb_dim: int = 256,
    ):
        super().__init__()

        # Generate channel attention weights from CB features
        self.gate_generator = nn.Sequential(
            nn.Linear(cb_dim, rgb_srm_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_srm_dim // 4, rgb_srm_dim),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, rgb_srm_features: torch.Tensor, cb_features: torch.Tensor) -> torch.Tensor:
        """
        Apply CB-guided channel attention to RGB-SRM features.

        Args:
            rgb_srm_features: (B, 2048, H, W) spatial features
            cb_features: (B, 256) global CB features

        Returns:
            gated_features: (B, 2048, H, W) attention-gated features
        """
        # Generate channel gates from CB features
        gates = self.gate_generator(cb_features)  # (B, 2048)
        gates = gates.unsqueeze(-1).unsqueeze(-1)  # (B, 2048, 1, 1)

        # Apply gating
        gated = rgb_srm_features * gates

        # Residual connection
        return rgb_srm_features + gated


if __name__ == '__main__':
    # Test fusion modules
    B, C, H, W = 2, 2048, 8, 8
    cb_dim = 256

    rgb_srm_feat = torch.randn(B, C, H, W)
    cb_feat = torch.randn(B, cb_dim)

    # Test ThreeStreamFusionModule
    fusion = ThreeStreamFusionModule()
    fused = fusion(rgb_srm_feat, cb_feat)
    print(f"ThreeStreamFusion output: {fused.shape}")  # Expected: (2, 2048, 8, 8)

    # Test ChannelAttentionFusion
    ca_fusion = ChannelAttentionFusion()
    gated = ca_fusion(rgb_srm_feat, cb_feat)
    print(f"ChannelAttentionFusion output: {gated.shape}")  # Expected: (2, 2048, 8, 8)
