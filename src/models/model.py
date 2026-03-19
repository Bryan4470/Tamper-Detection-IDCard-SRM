"""
RGB + Cb Dual-Stream Tamper Detection Model

Architecture:
1. RGB Stream: Pretrained CNN backbone for semantic understanding
2. Cb Stream: Patch-based color consistency detection
3. Cross-modal Fusion: Attention-based feature fusion
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict
from torch.hub import load_state_dict_from_url

from .cb_utils import extract_cb_channel, BackgroundRegionExtractor
from .srm_utils import SRMFilter


class CbEncoder(nn.Module):
    """Lightweight CNN encoder for Cb channel patches."""

    def __init__(self, output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, cb_patches: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = cb_patches.shape
        x = cb_patches.view(B * N, C, H, W)
        x = self.conv(x)
        x = x.view(B * N, -1)
        x = self.fc(x)
        return x.view(B, N, self.output_dim)


class SRMEncoder(nn.Module):
    """
    Lightweight CNN encoder for SRM noise residual maps.

    Input:  (B, 30, H, W)  — SRM noise maps from SRMFilter
    Output: (B, output_dim) — global noise feature vector
    """

    def __init__(self, output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(30, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, srm_map: torch.Tensor) -> torch.Tensor:
        x = self.conv(srm_map)
        x = x.flatten(1)
        return self.fc(x)


class CrossModalFusion(nn.Module):
    """Cross-modal fusion using attention mechanism."""

    def __init__(self, rgb_dim: int, cb_dim: int, hidden_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.rgb_proj = nn.Linear(rgb_dim, hidden_dim)
        self.cb_proj = nn.Linear(cb_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, rgb_global: torch.Tensor, cb_patch_features: torch.Tensor) -> torch.Tensor:
        rgb_embed = self.rgb_proj(rgb_global).unsqueeze(1)
        cb_embed = self.cb_proj(cb_patch_features)

        attn_out, _ = self.cross_attn(query=rgb_embed, key=cb_embed, value=cb_embed)
        attn_out = attn_out.squeeze(1)

        fused = torch.cat([rgb_embed.squeeze(1), attn_out], dim=1)
        return self.fusion(fused)


class RGBCbTamperDetector(nn.Module):
    """Dual-stream tamper detection model combining RGB and Cb streams."""

    BACKBONE_DIMS = {
        'resnet18': 512,
        'resnet50': 2048,
        'efficientnet_b0': 1280,
        'efficientnet_b3': 1536
    }

    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        num_classes: int = 2,
        cb_encoder_dim: int = 128,
        fusion_dim: int = 256,
        regions_config: str = None,
        patch_grid_size: int = 4,
        dropout: float = 0.5,
        use_srm: bool = False,
        srm_encoder_dim: int = 128,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.use_srm = use_srm

        if regions_config is None:
            regions_config = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'configs', 'regions.yaml'
            )

        if backbone not in self.BACKBONE_DIMS:
            raise ValueError(f"Unsupported backbone: {backbone}")

        rgb_feature_dim = self.BACKBONE_DIMS[backbone]

        # RGB Stream - Load pretrained weights
        # Create model without pretrained weights first, then manually load if needed
        if backbone == 'resnet18':
            base_model = models.resnet18(weights=None)
            if pretrained:
                weights_url = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
                state_dict = load_state_dict_from_url(weights_url, progress=True, check_hash=False)
                base_model.load_state_dict(state_dict)
        elif backbone == 'resnet50':
            base_model = models.resnet50(weights=None)
            if pretrained:
                weights_url = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
                state_dict = load_state_dict_from_url(weights_url, progress=True, check_hash=False)
                base_model.load_state_dict(state_dict)
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=None)
            if pretrained:
                weights_url = 'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth'
                state_dict = load_state_dict_from_url(weights_url, progress=True, check_hash=False)
                base_model.load_state_dict(state_dict)
        elif backbone == 'efficientnet_b3':
            base_model = models.efficientnet_b3(weights=None)
            if pretrained:
                weights_url = 'https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth'
                state_dict = load_state_dict_from_url(weights_url, progress=True, check_hash=False)
                base_model.load_state_dict(state_dict)

        self.rgb_backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.rgb_global_pool = nn.AdaptiveAvgPool2d(1)

        # Cb Stream
        self.bg_extractor = BackgroundRegionExtractor(regions_config, patch_grid_size)
        self.cb_encoder = CbEncoder(output_dim=cb_encoder_dim, dropout=dropout)

        # Fusion
        self.fusion = CrossModalFusion(rgb_feature_dim, cb_encoder_dim, fusion_dim, 4, dropout)

        # SRM stream (optional, config-gated)
        if use_srm:
            self.srm_filter = SRMFilter()          # fixed kernels, no grad
            self.srm_encoder = SRMEncoder(srm_encoder_dim, dropout)
            self.srm_proj = nn.Linear(srm_encoder_dim, fusion_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, images: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        # RGB Stream
        rgb_features = self.rgb_backbone(images)
        rgb_global = self.rgb_global_pool(rgb_features).flatten(1)

        # Cb Stream
        cb_channel = extract_cb_channel(images)
        cb_patches = self.bg_extractor(cb_channel)
        cb_features = self.cb_encoder(cb_patches)

        # Fusion
        fused_features = self.fusion(rgb_global, cb_features)

        # SRM stream (optional)
        if self.use_srm:
            srm_map = self.srm_filter(images)              # (B, 30, H, W)
            srm_global = self.srm_encoder(srm_map)         # (B, srm_encoder_dim)
            srm_embed = self.srm_proj(srm_global)          # (B, fusion_dim)
            fused_features = fused_features + srm_embed

        logits = self.classifier(fused_features)

        output = {'logits': logits}
        if return_features:
            output['cb_features'] = cb_features
            output['rgb_features'] = rgb_global
            output['fused_features'] = fused_features
            if self.use_srm:
                output['srm_features'] = srm_global

        return output

    def get_num_patches(self) -> int:
        return self.bg_extractor.get_num_patches()


def get_model(config: dict) -> RGBCbTamperDetector:
    """Factory function to create model from configuration."""
    model_config = config.get('model', {})
    regions_config = model_config.get('regions_config')

    if regions_config is None:
        regions_config = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'configs', 'regions.yaml'
        )

    return RGBCbTamperDetector(
        backbone=model_config.get('backbone', 'resnet18'),
        pretrained=model_config.get('pretrained', True),
        num_classes=model_config.get('num_classes', 2),
        cb_encoder_dim=model_config.get('cb_encoder_dim', 128),
        fusion_dim=model_config.get('fusion_dim', 256),
        regions_config=regions_config,
        patch_grid_size=model_config.get('patch_grid_size', 4),
        dropout=model_config.get('dropout', 0.5),
        use_srm=model_config.get('use_srm', False),
        srm_encoder_dim=model_config.get('srm_encoder_dim', 128),
    )
