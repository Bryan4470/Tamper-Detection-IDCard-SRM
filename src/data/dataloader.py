import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
import numpy as np
import pickle
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EKYCDataset(Dataset):
    """
    eKYC Dataset with aggressive data augmentation.

    Features:
    - Adds augmentation pipeline BEFORE standard transforms
    - Preserves all original functionality
    - Compatible with existing data directory structure
    """

    def __init__(self, image_paths, labels, transform=None, class_names=None,
                 aug=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels (0,1)
            transform: Standard torchvision transforms (applied AFTER aug)
            class_names: List of class names for reference
            aug: Augmentation pipeline (applied FIRST)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.aug = aug
        self.class_names = class_names or ['genuine', 'tamper']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image (already validated during _load_data)
        image = Image.open(self.image_paths[idx])

        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Get image dimensions for cropping
        w, h = image.size

        # Face region crop with percentage-based coordinates
        # Based on standard 1259x800 size where original crop was (0, 180, w, h)
        # Top crop ratio: 180/800 = 0.225 (22.5% from top)
        # NOTE: This matches the inference crop for face region
        # top_crop_ratio = 180 / 800  # 0.225 or 22.5%
        # crop_box = (0, int(h * top_crop_ratio), w, h)  # (left, top, right, bottom)
        # image = image.crop(crop_box)

        # STEP 1: Apply augmentation FIRST (if provided)
        # This simulates real-world quality variations
        if self.aug is not None:
            # Convert PIL Image to numpy array for albumentations
            image_np = np.array(image)
            # Apply augmentation with named argument
            augmented = self.aug(image=image_np)
            # Get the augmented image and convert back to PIL
            image = Image.fromarray(augmented['image'])

        # STEP 2: Apply standard torchvision transforms
        # This includes Resize, ToTensor, Normalize, etc.
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label


class EKYCDataLoader:
    """
    DataLoader with configurable augmentation.

    Accepts custom augmentation transforms to be applied during training.
    """

    def __init__(
        self,
        root_dir,
        image_size=224,
        batch_size=32,
        val_split=0.2,
        test_split=0.1,
        random_state=42,
        augmentation=None,
        num_workers=4,
        aug_config=None
    ):
        """
        Args:
            root_dir: Root directory containing class folders
            image_size: Input image size
            batch_size: Batch size for training
            val_split: Validation split ratio
            test_split: Test split ratio
            random_state: Random seed for reproducibility
            augmentation: Custom augmentation transform (None for no augmentation)
            num_workers: Number of data loading workers
            aug_config: Augmentation configuration dict from YAML (optional)
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        self.augmentation = augmentation
        self.num_workers = num_workers
        self.aug_config = aug_config

        # Class mapping
        self.class_names = ['genuine', 'tamper']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Data containers
        self.all_image_paths = []
        self.all_labels = []

        # Load data
        self._load_data()

        # Create transforms
        self._create_transforms()

        # Split data
        self._split_data()

    def _load_data(self):
        """Load image paths and labels from folder structure (with caching)"""
        cache_file = os.path.join(self.root_dir, "dataset_cache.pkl")

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                print(f"Loading dataset from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.all_image_paths = cache_data['image_paths']
                    self.all_labels = cache_data['labels']

                print(f"✅ Loaded {len(self.all_image_paths)} images from cache")
                self._print_class_distribution()
                return
            except Exception as e:
                print(f"⚠️  Cache loading failed ({e}), rebuilding dataset...")

        # Cache doesn't exist or failed to load - scan directories
        print("Loading dataset from directories (first time - will be cached)...")

        for class_name in self.class_names:
            class_folder = os.path.join(self.root_dir, class_name)

            if not os.path.exists(class_folder):
                print(f"Warning: Class folder {class_folder} not found!")
                continue

            print(f"Processing {class_name} folder...")

            # Find all CSV files in the folder
            csv_files = [f for f in os.listdir(class_folder) if f.endswith('.csv')]
            csv_images = set()  # Track images already loaded from CSV

            # Method 1: Load from CSV files if they exist
            for csv_file in csv_files:
                csv_path = os.path.join(class_folder, csv_file)
                print(f"  Loading from CSV: {csv_file}")

                try:
                    # Read CSV with explicit dtypes and prevent NaN conversion
                    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

                    # Check for image_path column
                    if 'image_path' in df.columns:
                        csv_count = 0
                        for _, row in df.iterrows():
                            img_path = row['image_path']

                            # Handle relative paths
                            if not os.path.isabs(img_path):
                                img_path = os.path.join(class_folder, img_path)

                            # Validate image before adding to dataset
                            if os.path.exists(img_path):
                                if self._validate_image(img_path):
                                    self.all_image_paths.append(img_path)
                                    self.all_labels.append(self.class_to_idx[class_name])
                                    csv_images.add(os.path.basename(img_path))
                                    csv_count += 1
                                else:
                                    logging.warning(f"Skipping corrupted image during loading: {img_path}")
                            else:
                                logging.warning(f"Image not found: {img_path}")

                        print(f"    Loaded {csv_count} images from {csv_file}")
                    else:
                        print(f"    Warning: 'image_path' column not found in {csv_file}")

                except Exception as e:
                    print(f"    Error reading {csv_file}: {e}")

        print(f"\nTotal images loaded: {len(self.all_image_paths)}")
        self._print_class_distribution()

        # Save to cache for faster loading next time
        try:
            cache_data = {
                'image_paths': self.all_image_paths,
                'labels': self.all_labels
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Saved dataset cache to {cache_file}")
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")

    def _validate_image(self, img_path):
        """Validate that an image can be loaded properly"""
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify it's a valid image
            # Re-open after verify (verify closes the file)
            with Image.open(img_path) as img:
                img.load()  # Actually load the image data
            return True
        except Exception as e:
            return False

    def _print_class_distribution(self):
        """Print class distribution"""
        class_counts = Counter(self.all_labels)
        print("\nClass Distribution:")
        print("-" * 30)
        for idx, class_name in enumerate(self.class_names):
            count = class_counts.get(idx, 0)
            percentage = (count / len(self.all_labels)) * 100 if self.all_labels else 0
            print(f"{class_name}: {count} images ({percentage:.1f}%)")

    def _create_transforms(self):
        """Create standard data transforms (applied AFTER augmentation)"""
        from .augmentation import build_train_transform, build_val_transform

        # Get augmentation config if provided
        aug_config = getattr(self, 'aug_config', None)

        # Build training transform with config-based augmentation
        self.train_transform = build_train_transform(
            image_size=self.image_size,
            aug_config=aug_config,
            normalize=True
        )

        # Build validation/test transform (no augmentation)
        self.val_transform = build_val_transform(
            image_size=self.image_size,
            normalize=True
        )

    def _split_data(self):
        """Split data into train, validation, and test sets"""
        splits_cache = os.path.join(self.root_dir, "splits.pkl")

        if os.path.exists(splits_cache):
            print(f"\nLoading existing splits from {splits_cache}")
            with open(splits_cache, 'rb') as f:
                splits = pickle.load(f)
                train_indices = splits['train']
                val_indices = splits['val']
                test_indices = splits['test']
        else:
            print("\nCreating new train/val/test splits...")

            # Handle test split
            if self.test_split > 0:
                # First split: separate test set
                train_val_indices, test_indices = train_test_split(
                    range(len(self.all_image_paths)),
                    test_size=self.test_split,
                    random_state=self.random_state,
                    stratify=self.all_labels
                )
            else:
                # No test split, use all data for train/val
                train_val_indices = list(range(len(self.all_image_paths)))
                test_indices = []

            # Second split: separate train and validation
            train_labels_subset = [self.all_labels[i] for i in train_val_indices]
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=self.val_split / (1 - self.test_split) if self.test_split > 0 else self.val_split,
                random_state=self.random_state,
                stratify=train_labels_subset
            )

            # Save splits for reproducibility
            splits = {
                'train': train_indices,
                'val': val_indices,
                'test': test_indices
            }
            with open(splits_cache, 'wb') as f:
                pickle.dump(splits, f)
            print(f"Saved splits to {splits_cache}")

        # Create datasets
        train_paths = [self.all_image_paths[i] for i in train_indices]
        train_labels = [self.all_labels[i] for i in train_indices]

        val_paths = [self.all_image_paths[i] for i in val_indices]
        val_labels = [self.all_labels[i] for i in val_indices]

        test_paths = [self.all_image_paths[i] for i in test_indices]
        test_labels = [self.all_labels[i] for i in test_indices]

        # Create datasets with custom augmentation for training only
        print(f"\nAugmentation: {'Enabled' if self.augmentation is not None else 'Disabled'}")

        self.train_dataset = EKYCDataset(
            train_paths, train_labels,
            transform=self.train_transform,
            class_names=self.class_names,
            aug=self.augmentation  # Custom augmentation ONLY for training
        )

        self.val_dataset = EKYCDataset(
            val_paths, val_labels,
            transform=self.val_transform,
            class_names=self.class_names,
            aug=None  # NO augmentation for validation
        )

        self.test_dataset = EKYCDataset(
            test_paths, test_labels,
            transform=self.val_transform,
            class_names=self.class_names,
            aug=None  # NO augmentation for test
        )

        print(f"\nDataset splits:")
        print(f"  Training:   {len(self.train_dataset)} images")
        print(f"  Validation: {len(self.val_dataset)} images")
        print(f"  Test:       {len(self.test_dataset)} images")

    def get_dataloaders(self):
        """Get train, validation, and test dataloaders"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader
