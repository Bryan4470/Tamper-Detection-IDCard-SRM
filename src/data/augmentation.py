"""
Augmentation utilities for building transforms from configuration

This module provides functions to create augmentation pipelines
from YAML configuration files, allowing flexible experimentation
without code changes.
"""

from torchvision import transforms
from typing import Dict, Any, List


def build_augmentation_from_config(aug_config: Dict[str, Any]) -> transforms.Compose:
    """
    Build augmentation pipeline from configuration dictionary.

    Args:
        aug_config: Augmentation configuration from YAML file

    Returns:
        transforms.Compose object with configured augmentations
    """
    aug_list = []

    # 1. Random Horizontal Flip
    if aug_config.get('random_horizontal_flip', {}).get('enabled', False):
        prob = aug_config['random_horizontal_flip'].get('probability', 0.5)
        aug_list.append(transforms.RandomHorizontalFlip(p=prob))
        print(f"  ✓ RandomHorizontalFlip(p={prob})")

    # 2. Random Rotation
    if aug_config.get('random_rotation', {}).get('enabled', False):
        degrees = aug_config['random_rotation'].get('degrees', 5)
        aug_list.append(transforms.RandomRotation(degrees=degrees))
        print(f"  ✓ RandomRotation(degrees=±{degrees})")

    # 3. Random Affine (more advanced than just rotation)
    if aug_config.get('random_affine', {}).get('enabled', False):
        affine_cfg = aug_config['random_affine']
        degrees = affine_cfg.get('degrees', 0)
        translate = affine_cfg.get('translate', None)
        scale = affine_cfg.get('scale', None)
        shear = affine_cfg.get('shear', None)

        aug_list.append(transforms.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear
        ))
        print(f"  ✓ RandomAffine(degrees={degrees}, translate={translate}, scale={scale}, shear={shear})")

    # 4. Color Jitter
    if aug_config.get('color_jitter', {}).get('enabled', False):
        jitter_cfg = aug_config['color_jitter']
        brightness = jitter_cfg.get('brightness', 0.2)
        contrast = jitter_cfg.get('contrast', 0.2)
        saturation = jitter_cfg.get('saturation', 0.2)
        hue = jitter_cfg.get('hue', 0.1)

        aug_list.append(transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ))
        print(f"  ✓ ColorJitter(brightness={brightness}, contrast={contrast}, saturation={saturation}, hue={hue})")

    # 5. Gaussian Blur
    if aug_config.get('gaussian_blur', {}).get('enabled', False):
        blur_cfg = aug_config['gaussian_blur']
        kernel_size = blur_cfg.get('kernel_size', 3)
        probability = blur_cfg.get('probability', 0.1)

        aug_list.append(transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=kernel_size)],
            p=probability
        ))
        print(f"  ✓ GaussianBlur(kernel_size={kernel_size}, p={probability})")

    # 6. Random Adjust Sharpness
    if aug_config.get('random_adjust_sharpness', {}).get('enabled', False):
        sharp_cfg = aug_config['random_adjust_sharpness']
        sharpness_factor = sharp_cfg.get('sharpness_factor', 2.0)
        probability = sharp_cfg.get('probability', 0.1)

        aug_list.append(transforms.RandomApply(
            [transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor)],
            p=probability
        ))
        print(f"  ✓ RandomAdjustSharpness(factor={sharpness_factor}, p={probability})")

    # 7. Random Erasing (use with caution for tampering detection!)
    # Note: This is applied AFTER ToTensor, so we'll add it separately in dataloader
    # Just mark if it should be used
    if aug_config.get('random_erasing', {}).get('enabled', False):
        print(f"  ⚠️  RandomErasing enabled (will be applied after ToTensor)")

    # 8. Random Grayscale
    if aug_config.get('random_grayscale', {}).get('enabled', False):
        probability = aug_config['random_grayscale'].get('probability', 0.1)
        aug_list.append(transforms.RandomGrayscale(p=probability))
        print(f"  ✓ RandomGrayscale(p={probability})")

    if not aug_list:
        print("  ⚠️  No augmentations enabled in config")
        return None

    # Return composed transforms (without ToTensor and Normalize)
    # Those will be added by the dataloader
    return transforms.Compose(aug_list)


def build_train_transform(
    image_size: int,
    aug_config: Dict[str, Any] = None,
    normalize: bool = True
) -> transforms.Compose:
    """
    Build complete training transform pipeline.

    Args:
        image_size: Target image size
        aug_config: Augmentation configuration (optional)
        normalize: Whether to normalize with ImageNet stats

    Returns:
        Complete transform pipeline for training
    """
    transform_list = [transforms.Resize((image_size, image_size))]

    # Add augmentations if provided
    if aug_config:
        print("\nBuilding augmentation pipeline:")
        aug_transforms = build_augmentation_from_config(aug_config)
        if aug_transforms is not None:
            transform_list.extend(aug_transforms.transforms)

    # Add ToTensor
    transform_list.append(transforms.ToTensor())

    # Add RandomErasing if enabled (must be after ToTensor)
    if aug_config and aug_config.get('random_erasing', {}).get('enabled', False):
        erasing_cfg = aug_config['random_erasing']
        probability = erasing_cfg.get('probability', 0.1)
        scale = tuple(erasing_cfg.get('scale', [0.02, 0.1]))
        ratio = tuple(erasing_cfg.get('ratio', [0.3, 3.3]))

        transform_list.append(transforms.RandomErasing(
            p=probability,
            scale=scale,
            ratio=ratio
        ))
        print(f"  ✓ RandomErasing(p={probability}, scale={scale}, ratio={ratio})")

    # Add normalization
    if normalize:
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))

    return transforms.Compose(transform_list)


def build_val_transform(image_size: int, normalize: bool = True) -> transforms.Compose:
    """
    Build validation/test transform pipeline (no augmentation).

    Args:
        image_size: Target image size
        normalize: Whether to normalize with ImageNet stats

    Returns:
        Transform pipeline for validation/testing
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]

    if normalize:
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))

    return transforms.Compose(transform_list)


def print_augmentation_summary(aug_config: Dict[str, Any]):
    """
    Print a summary of augmentation configuration.

    Args:
        aug_config: Augmentation configuration
    """

    print("AUGMENTATION CONFIGURATION")
    print("="*70)

    enabled_augs = []
    disabled_augs = []

    aug_types = [
        'random_horizontal_flip',
        'random_rotation',
        'random_affine',
        'color_jitter',
        'gaussian_blur',
        'random_adjust_sharpness',
        'random_erasing',
        'random_grayscale'
    ]

    for aug_type in aug_types:
        if aug_config.get(aug_type, {}).get('enabled', False):
            enabled_augs.append(aug_type)
        else:
            disabled_augs.append(aug_type)

    print(f"\nEnabled ({len(enabled_augs)}):")
    for aug in enabled_augs:
        print(f"  ✅ {aug}")

    print(f"\nDisabled ({len(disabled_augs)}):")
    for aug in disabled_augs:
        print(f"  ⛔ {aug}")

    print("="*70)


if __name__ == '__main__':
    # Test augmentation building
    import yaml

    print("Testing augmentation builder...")

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    aug_config = config.get('augmentation', {})

    # Print summary
    print_augmentation_summary(aug_config)

    # Build transforms
    train_transform = build_train_transform(224, aug_config)
    val_transform = build_val_transform(224)

    print("\n✅ Augmentation pipeline built successfully!")
    print(f"\nTrain transforms ({len(train_transform.transforms)} steps):")
    for i, t in enumerate(train_transform.transforms):
        print(f"  {i+1}. {t.__class__.__name__}")

    print(f"\nVal transforms ({len(val_transform.transforms)} steps):")
    for i, t in enumerate(val_transform.transforms):
        print(f"  {i+1}. {t.__class__.__name__}")
