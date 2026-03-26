import torch
import torch.nn as nn
import torch.nn.functional as F

from components.attention import ChannelAttention, SpatialAttention, DualCrossModalAttention
from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from networks.xception import TransferModel
from networks.convnext import ConvNeXtTransferModel, CONVNEXT_DIMS


class SRMPixelAttention(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(SRMPixelAttention, self).__init__()
        self.srm = SRMConv2d_simple()
        # Intermediate channels scale with output
        mid_channels = out_channels // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 2, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.pa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
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
    """
    Two-Stream Network for tamper detection.

    Args:
        backbone: Backbone architecture to use ('xception' or 'convnext')
        dropout: Dropout rate for classifier
    """

    # Feature dimension configurations for different backbones
    BACKBONE_CONFIGS = {
        'xception': {
            'part1_0': 32,   # After first conv
            'part1_1': 64,   # After second conv
            'part2': 728,    # After blocks 1-3 (attention dim)
            'part3': 728,    # After blocks 4-7 (attention dim)
            'part4': 728,    # After blocks 8-11
            'part5': 2048,   # Exit flow output
            'srm_sa_channels': 64,  # Channels for SRM spatial attention post
        },
        'convnext': {
            'part1_0': 96,   # After stem
            'part1_1': 96,   # After stage0
            'part2': 192,    # After stage1 (attention dim)
            'part3': 384,    # After stage2 (attention dim)
            'part4': 768,    # After stage3
            'part5': 2048,   # Output projection
            'srm_sa_channels': 96,  # Channels for SRM spatial attention post
        },
    }

    def __init__(self, backbone='xception', dropout=0.5):
        super().__init__()
        self.backbone_name = backbone

        # Get backbone configuration
        if backbone not in self.BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(self.BACKBONE_CONFIGS.keys())}")
        config = self.BACKBONE_CONFIGS[backbone]

        # Create backbone streams
        if backbone == 'xception':
            self.backbone_rgb = TransferModel(
                'xception', dropout=dropout, inc=3, return_fea=True)
            self.backbone_srm = TransferModel(
                'xception', dropout=dropout, inc=3, return_fea=True)
        elif backbone == 'convnext':
            self.backbone_rgb = ConvNeXtTransferModel(
                dropout=dropout, inc=3, return_fea=True)
            self.backbone_srm = ConvNeXtTransferModel(
                dropout=dropout, inc=3, return_fea=True)

        # SRM convolutions - dimensions depend on backbone
        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(config['part1_0'], config['part1_0'])
        self.srm_conv2 = SRMConv2d_Separate(config['part1_1'], config['part1_1'])
        self.relu = nn.ReLU(inplace=True)

        self.att_map = None
        self.srm_sa = SRMPixelAttention(3, out_channels=config['srm_sa_channels'])
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(config['srm_sa_channels']),
            nn.ReLU(inplace=True)
        )

        # Dual cross-modal attention - dimensions depend on backbone
        # Note: size=None enables dynamic spatial dimension handling
        self.dual_cma0 = DualCrossModalAttention(in_dim=config['part2'], ret_att=False)
        self.dual_cma1 = DualCrossModalAttention(in_dim=config['part3'], ret_att=False)

        self.fusion = FeatureFusionModule()

        self.att_dic = {}

    @property
    def _rgb_model(self):
        """Get the underlying RGB backbone model."""
        if self.backbone_name == 'xception':
            return self.backbone_rgb.model
        elif self.backbone_name == 'convnext':
            return self.backbone_rgb.model

    @property
    def _srm_model(self):
        """Get the underlying SRM backbone model."""
        if self.backbone_name == 'xception':
            return self.backbone_srm.model
        elif self.backbone_name == 'convnext':
            return self.backbone_srm.model

    def features(self, x):
        srm = self.srm_conv0(x)

        rgb_model = self._rgb_model
        srm_model = self._srm_model

        x = rgb_model.fea_part1_0(x)
        y = srm_model.fea_part1_0(srm) \
            + self.srm_conv1(x)
        y = self.relu(y)

        x = rgb_model.fea_part1_1(x)
        y = srm_model.fea_part1_1(y) \
            + self.srm_conv2(x)
        y = self.relu(y)

        # srm guided spatial attention
        # Resize attention map to match feature spatial dimensions
        att_map = self.srm_sa(srm)
        if att_map.shape[2:] != x.shape[2:]:
            att_map = F.interpolate(att_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        self.att_map = att_map
        x = x * self.att_map + x
        x = self.srm_sa_post(x)

        x = rgb_model.fea_part2(x)
        y = srm_model.fea_part2(y)

        x, y = self.dual_cma0(x, y)

        x = rgb_model.fea_part3(x)
        y = srm_model.fea_part3(y)

        x, y = self.dual_cma1(x, y)

        x = rgb_model.fea_part4(x)
        y = srm_model.fea_part4(y)

        x = rgb_model.fea_part5(x)
        y = srm_model.fea_part5(y)

        fea = self.fusion(x, y)

        return fea

    def classifier(self, fea):
        out, fea = self.backbone_rgb.classifier(fea)
        return out, fea

    def forward(self, x):
        '''
        x: original rgb
        '''
        out, fea = self.classifier(self.features(x))

        return out, fea, self.att_map
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['xception', 'convnext'],
                        help='Backbone architecture to use')
    args = parser.parse_args()

    print(f"\nTesting Two_Stream_Net with backbone: {args.backbone}")
    print("=" * 60)

    model = Two_Stream_Net(backbone=args.backbone)
    dummy = torch.rand((1, 3, 256, 256))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Forward pass
    print("\nRunning forward pass...")
    out, fea, att_map = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Feature shape: {fea.shape}")
    print(f"Attention map shape: {att_map.shape}")

    print("\nModel architecture:")
    print(model)
    