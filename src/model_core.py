import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from components.attention import ChannelAttention, SpatialAttention, DualCrossModalAttention
from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from components.cb_stream import CbStream
from components.fusion import ThreeStreamFusionModule
from networks.xception import TransferModel


class SRMPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SRMPixelAttention, self).__init__()
        self.srm = SRMConv2d_simple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.pa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_srm = self.srm(x)
        fea = self.conv(x_srm)        
        att_map = self.pa(fea)
        
        return att_map


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048*2, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Two_Stream_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)

        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)
        self.relu = nn.ReLU(inplace=True)

        self.att_map = None
        self.srm_sa = SRMPixelAttention(3)
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.dual_cma1 = DualCrossModalAttention(in_dim=728, ret_att=False)

        self.fusion = FeatureFusionModule()

        self.att_dic = {}

    def features(self, x):
        srm = self.srm_conv0(x)

        x = self.xception_rgb.model.fea_part1_0(x)
        y = self.xception_srm.model.fea_part1_0(srm) \
            + self.srm_conv1(x)
        y = self.relu(y)

        x = self.xception_rgb.model.fea_part1_1(x)
        y = self.xception_srm.model.fea_part1_1(y) \
            + self.srm_conv2(x)
        y = self.relu(y)

        # srm guided spatial attention
        self.att_map = self.srm_sa(srm)
        x = x * self.att_map + x
        x = self.srm_sa_post(x)

        x = self.xception_rgb.model.fea_part2(x)
        y = self.xception_srm.model.fea_part2(y)

        x, y = self.dual_cma0(x, y)


        x = self.xception_rgb.model.fea_part3(x)        
        y = self.xception_srm.model.fea_part3(y)
 

        x, y = self.dual_cma1(x, y)

        x = self.xception_rgb.model.fea_part4(x)
        y = self.xception_srm.model.fea_part4(y)

        x = self.xception_rgb.model.fea_part5(x)
        y = self.xception_srm.model.fea_part5(y)

        fea = self.fusion(x, y)
                

        return fea

    def classifier(self, fea):
        out, fea = self.xception_rgb.classifier(fea)
        return out, fea

    def forward(self, x):
        '''
        x: original rgb
        '''
        out, fea = self.classifier(self.features(x))

        return out, fea, self.att_map
    
class Three_Stream_Net(nn.Module):
    """
    Three-Stream Network: RGB + SRM + CB for ID Card Tamper Detection.

    Extends Two_Stream_Net by adding:
    1. CbStream: Extracts Cb channel features from ID card regions
    2. ThreeStreamFusionModule: Fuses RGB-SRM and CB features

    Architecture:
    - RGB Stream (Xception backbone)
    - SRM Stream (Xception backbone with SRM filters)
    - CB Stream (Lightweight CNN on Cb channel patches)
    - DualCrossModalAttention between RGB and SRM
    - ThreeStreamFusion for final feature integration
    """

    def __init__(
        self,
        regions_config: Optional[str] = None,
        cb_output_dim: int = 256,
        patch_grid_size: int = 4,
        fixed_region_size: int = 64,
    ):
        """
        Args:
            regions_config: Path to regions YAML config. If None, uses default path.
            cb_output_dim: Output dimension of CB stream (default 256)
            patch_grid_size: Grid size for CB patches (4 = 4x4 = 16 patches)
            fixed_region_size: Fixed size for each region (64 for 256x256 input)
        """
        super().__init__()

        # ─── RGB and SRM Streams (same as Two_Stream_Net) ───
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)

        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)
        self.relu = nn.ReLU(inplace=True)

        self.att_map = None
        self.srm_sa = SRMPixelAttention(3)
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.dual_cma1 = DualCrossModalAttention(in_dim=728, ret_att=False)

        # RGB-SRM Fusion (same as Two_Stream_Net)
        self.fusion = FeatureFusionModule()

        # ─── CB Stream (NEW) ───
        if regions_config is None:
            # Default path relative to this file
            regions_config = os.path.join(
                os.path.dirname(__file__), '..', 'configs', 'regions.yaml'
            )

        self.cb_stream = CbStream(
            regions_config=regions_config,
            patch_grid_size=patch_grid_size,
            fixed_region_size=fixed_region_size,
            encoder_dim=128,
            output_dim=cb_output_dim,
        )

        # ─── Three-Stream Fusion (NEW) ───
        self.three_stream_fusion = ThreeStreamFusionModule(
            rgb_srm_dim=2048,
            cb_dim=cb_output_dim,
            output_dim=2048,
        )

        self.att_dic = {}

    def rgb_srm_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract RGB-SRM fused features (same as Two_Stream_Net.features).

        Args:
            x: Input RGB tensor (B, 3, H, W)

        Returns:
            Fused features (B, 2048, 8, 8)
        """
        srm = self.srm_conv0(x)

        x_rgb = self.xception_rgb.model.fea_part1_0(x)
        y = self.xception_srm.model.fea_part1_0(srm) + self.srm_conv1(x_rgb)
        y = self.relu(y)

        x_rgb = self.xception_rgb.model.fea_part1_1(x_rgb)
        y = self.xception_srm.model.fea_part1_1(y) + self.srm_conv2(x_rgb)
        y = self.relu(y)

        # SRM guided spatial attention
        self.att_map = self.srm_sa(srm)
        x_rgb = x_rgb * self.att_map + x_rgb
        x_rgb = self.srm_sa_post(x_rgb)

        x_rgb = self.xception_rgb.model.fea_part2(x_rgb)
        y = self.xception_srm.model.fea_part2(y)

        x_rgb, y = self.dual_cma0(x_rgb, y)

        x_rgb = self.xception_rgb.model.fea_part3(x_rgb)
        y = self.xception_srm.model.fea_part3(y)

        x_rgb, y = self.dual_cma1(x_rgb, y)

        x_rgb = self.xception_rgb.model.fea_part4(x_rgb)
        y = self.xception_srm.model.fea_part4(y)

        x_rgb = self.xception_rgb.model.fea_part5(x_rgb)
        y = self.xception_srm.model.fea_part5(y)

        fea = self.fusion(x_rgb, y)
        return fea

    def features(self, x: torch.Tensor, return_cb_features: bool = False):
        """
        Extract final features combining all three streams.

        Args:
            x: Input RGB tensor (B, 3, H, W) normalized to [-1, 1]
            return_cb_features: If True, also return CB patch features for loss

        Returns:
            fused_features: (B, 2048, 8, 8) final fused features
            cb_patch_features: (B, N, 128) if return_cb_features=True
        """
        # Get RGB-SRM features
        rgb_srm_fea = self.rgb_srm_features(x)

        # Get CB features
        if return_cb_features:
            cb_fea, cb_patch_fea = self.cb_stream(x, return_patch_features=True)
        else:
            cb_fea = self.cb_stream(x, return_patch_features=False)

        # Fuse all three streams
        fused_fea = self.three_stream_fusion(rgb_srm_fea, cb_fea)

        if return_cb_features:
            return fused_fea, cb_patch_fea

        return fused_fea

    def classifier(self, fea: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classification head (reuses RGB stream's classifier)."""
        out, fea = self.xception_rgb.classifier(fea)
        return out, fea

    def forward(
        self,
        x: torch.Tensor,
        return_cb_features: bool = False,
    ):
        """
        Forward pass.

        Args:
            x: Input RGB tensor (B, 3, H, W) normalized to [-1, 1]
            return_cb_features: If True, also return CB patch features for loss

        Returns:
            out: (B, num_classes) classification logits
            fea: (B, feature_dim) final features before classifier
            att_map: Attention map from SRM spatial attention
            cb_patch_features: (B, N, 128) if return_cb_features=True
        """
        if return_cb_features:
            fused_fea, cb_patch_fea = self.features(x, return_cb_features=True)
            out, fea = self.classifier(fused_fea)
            return out, fea, self.att_map, cb_patch_fea
        else:
            fused_fea = self.features(x, return_cb_features=False)
            out, fea = self.classifier(fused_fea)
            return out, fea, self.att_map

    @classmethod
    def from_two_stream(
        cls,
        checkpoint_path: str,
        regions_config: Optional[str] = None,
        strict: bool = False,
    ) -> 'Three_Stream_Net':
        """
        Load a Three_Stream_Net from a Two_Stream_Net checkpoint.

        Initializes RGB-SRM components from checkpoint, CB stream from scratch.

        Args:
            checkpoint_path: Path to Two_Stream_Net checkpoint
            regions_config: Path to regions YAML config
            strict: If True, raise error on missing keys

        Returns:
            Three_Stream_Net with RGB-SRM weights loaded
        """
        model = cls(regions_config=regions_config)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt

        # Filter out keys that don't exist in current model
        model_state = model.state_dict()
        filtered_state = {}
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v

        # Load weights
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)

        print(f"Loaded {len(filtered_state)} weights from Two_Stream_Net checkpoint")
        if missing:
            cb_missing = [k for k in missing if 'cb_stream' in k or 'three_stream_fusion' in k]
            other_missing = [k for k in missing if k not in cb_missing]
            print(f"  CB/Fusion components initialized from scratch: {len(cb_missing)} weights")
            if other_missing and strict:
                print(f"  WARNING: Missing from RGB-SRM: {other_missing}")

        return model

    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze or unfreeze RGB-SRM backbone weights.

        Used for Phase 1 training where only CB stream + fusion are trained.
        """
        for name, param in self.named_parameters():
            # Skip CB stream and three-stream fusion
            if 'cb_stream' in name or 'three_stream_fusion' in name:
                continue
            param.requires_grad = not freeze

        status = "frozen" if freeze else "unfrozen"
        print(f"RGB-SRM backbone {status}. CB stream and fusion remain trainable.")


if __name__ == '__main__':
    # Test Two_Stream_Net
    print("Testing Two_Stream_Net...")
    model = Two_Stream_Net()
    dummy = torch.rand((1, 3, 256, 256))
    out, fea, att = model(dummy)
    print(f"  Output: {out.shape}, Features: {fea.shape}")

    # Test Three_Stream_Net
    print("\nTesting Three_Stream_Net...")
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'regions.yaml')
    if os.path.exists(config_path):
        model3 = Three_Stream_Net(regions_config=config_path)
        out, fea, att = model3(dummy)
        print(f"  Output: {out.shape}, Features: {fea.shape}")

        # Test with CB features
        out, fea, att, cb_fea = model3(dummy, return_cb_features=True)
        print(f"  CB Features: {cb_fea.shape}")

        # Test backbone freezing
        model3.freeze_backbone(freeze=True)
        trainable = sum(p.numel() for p in model3.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model3.parameters())
        print(f"  Trainable params after freeze: {trainable:,} / {total:,}")
    else:
        print(f"  Skipping (regions config not found at {config_path})")
    