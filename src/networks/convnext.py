"""
ConvNeXt backbone wrapper for ID Card Tamper Detection.
Uses timm library for pretrained ConvNeXt-Tiny model.
Provides the same interface as the Xception backbone (fea_part1_0, fea_part1_1, etc.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first (NCHW) format."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt-Tiny backbone with Xception-compatible interface.
    Uses timm library for pretrained ImageNet weights.

    ConvNeXt stages (from timm):
    - stem: 4x4 conv, stride 4 -> 96 channels (H/4, W/4)
    - stages[0]: 96 -> 96 channels (3 blocks)
    - stages[1]: 96 -> 192 channels (3 blocks, downsample)
    - stages[2]: 192 -> 384 channels (9 blocks, downsample)
    - stages[3]: 384 -> 768 channels (3 blocks, downsample)

    Mapped to Xception interface:
    - fea_part1_0: stem (3 -> 96, stride 4)
    - fea_part1_1: stages[0] (96 -> 96)
    - fea_part2: stages[1] (96 -> 192, downsample)
    - fea_part3: stages[2] (192 -> 384, downsample)
    - fea_part4: stages[3] (384 -> 768, downsample)
    - fea_part5: output projection (768 -> 2048)
    """

    def __init__(self, pretrained=True, inc=3, num_classes=2, dropout=0.5, drop_path_rate=0.1):
        super(ConvNeXtBackbone, self).__init__()

        # Load ConvNeXt-Tiny from timm with pretrained ImageNet weights
        convnext = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            in_chans=inc,
            drop_path_rate=drop_path_rate,
            num_classes=0,  # Remove classifier head
        )

        if pretrained:
            print("[INFO] Loaded pretrained ConvNeXt-Tiny weights from timm (ImageNet-1K)")

        # Part 1_0: Stem (3 -> 96, stride 4)
        self.fea_part1_0 = convnext.stem

        # Part 1_1: Stage 0 (96 -> 96, no downsampling)
        self.fea_part1_1 = convnext.stages[0]

        # Part 2: Stage 1 (96 -> 192, includes downsample)
        self.fea_part2 = convnext.stages[1]

        # Part 3: Stage 2 (192 -> 384, includes downsample)
        self.fea_part3 = convnext.stages[2]

        # Part 4: Stage 3 (384 -> 768, includes downsample)
        self.fea_part4 = convnext.stages[3]

        # Part 5: Output projection (768 -> 2048) for fusion compatibility
        self.fea_part5 = nn.Sequential(
            LayerNorm2d(768, eps=1e-6),
            nn.Conv2d(768, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048),
        )

        # Classifier (matching Xception interface)
        self.relu = nn.ReLU(inplace=True)
        self.last_linear = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes)
        )

        # Initialize the new layers
        self._init_weights()

    def _init_weights(self):
        for m in [self.fea_part5, self.last_linear]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def features(self, x):
        x = self.fea_part1_0(x)
        x = self.fea_part1_1(x)
        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)
        x = self.fea_part5(x)
        return x

    def classifier(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.last_linear(x)
        return out, x

    def forward(self, x):
        x = self.features(x)
        out, fea = self.classifier(x)
        return out, fea


class ConvNeXtTransferModel(nn.Module):
    """
    Transfer learning wrapper for ConvNeXt, matching TransferModel interface.
    """

    def __init__(self, num_out_classes=2, dropout=0.5, inc=3, return_fea=False):
        super(ConvNeXtTransferModel, self).__init__()
        self.return_fea = return_fea
        self.model = ConvNeXtBackbone(
            pretrained=True,
            inc=inc,
            num_classes=num_out_classes,
            dropout=dropout
        )

    def forward(self, x):
        out, fea = self.model(x)
        if self.return_fea:
            return out, fea
        return out

    def features(self, x):
        return self.model.features(x)

    def classifier(self, x):
        return self.model.classifier(x)


# Feature dimension constants for ConvNeXt-Tiny
CONVNEXT_DIMS = {
    'part1_0': 96,   # After stem
    'part1_1': 96,   # After stage0
    'part2': 192,    # After stage1
    'part3': 384,    # After stage2 (attention applied here)
    'part4': 768,    # After stage3
    'part5': 2048,   # After output projection
}


if __name__ == '__main__':
    # Test the backbone
    model = ConvNeXtBackbone(pretrained=True)
    dummy = torch.rand(2, 3, 256, 256)

    print("Testing ConvNeXtBackbone...")
    print(f"Input shape: {dummy.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test each part
    x = model.fea_part1_0(dummy)
    print(f"After fea_part1_0: {x.shape}")  # Expected: (2, 96, 64, 64)

    x = model.fea_part1_1(x)
    print(f"After fea_part1_1: {x.shape}")  # Expected: (2, 96, 64, 64)

    x = model.fea_part2(x)
    print(f"After fea_part2: {x.shape}")    # Expected: (2, 192, 32, 32)

    x = model.fea_part3(x)
    print(f"After fea_part3: {x.shape}")    # Expected: (2, 384, 16, 16)

    x = model.fea_part4(x)
    print(f"After fea_part4: {x.shape}")    # Expected: (2, 768, 8, 8)

    x = model.fea_part5(x)
    print(f"After fea_part5: {x.shape}")    # Expected: (2, 2048, 8, 8)

    # Test full forward pass
    out, fea = model(dummy)
    print(f"Output shape: {out.shape}")     # Expected: (2, 2)
    print(f"Feature shape: {fea.shape}")    # Expected: (2, 2048)

    print("\nConvNeXtTransferModel test...")
    transfer_model = ConvNeXtTransferModel(num_out_classes=2, dropout=0.5, return_fea=True)
    out, fea = transfer_model(dummy)
    print(f"TransferModel output: {out.shape}, features: {fea.shape}")
