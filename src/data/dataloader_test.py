import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
import os
import pickle
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TestDatasetFromCSV(Dataset):
    """
    Test dataset that loads images from CSV file(s).

    CSV format:
        - image_path: absolute or relative path to image
        - fraud_type: 'genuine' or 'tamper'
    """

    def __init__(self, image_paths, labels, transform=None, class_names=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels (0=genuine, 1=tamper)
            transform: Transforms to apply to images
            class_names: List of class names
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or ['genuine', 'tamper']
        self.corrupted_images = set()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Handle corrupted images by skipping to next valid image
        max_attempts = len(self.image_paths)
        attempts = 0

        while attempts < max_attempts:
            current_idx = (idx + attempts) % len(self.image_paths)

            # Skip known corrupted images
            if self.image_paths[current_idx] in self.corrupted_images:
                attempts += 1
                continue

            # Try to load image
            try:
                image = Image.open(self.image_paths[current_idx])

                # Convert to RGB if not already
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Verify image is valid
                image.load()

                idx = current_idx
                break

            except Exception as e:
                logging.warning(f"Skipping corrupted image {self.image_paths[current_idx]}: {e}")
                self.corrupted_images.add(self.image_paths[current_idx])
                attempts += 1
        else:
            raise RuntimeError(f"Unable to load any valid images after {max_attempts} attempts")

        # Get image dimensions for cropping
        w, h = image.size

        # Face region crop with percentage-based coordinates
        # Based on standard 1259x800 size where original crop was (0, 180, w, h)
        # Top crop ratio: 180/800 = 0.225 (22.5% from top)
        # NOTE: This matches the training crop
        # top_crop_ratio = 180 / 800  # 0.225 or 22.5%
        # crop_box = (0, int(h * top_crop_ratio), w, h)  # (left, top, right, bottom)
        # image = image.crop(crop_box)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label


def load_test_dataset_from_csv(
    csv_paths,
    batch_size=32,
    image_size=224,
    num_workers=4,
    return_individual=False,
    use_cache=True
):
    """
    Load test dataset from one or more CSV files.

    Args:
        csv_paths: List of CSV file paths or single CSV path
        batch_size: Batch size for testing
        image_size: Input image size
        num_workers: Number of data loading workers
        return_individual: If True, return individual loaders for each CSV
        use_cache: If True, use cached data if available (default: True)

    Returns:
        If return_individual=False:
            test_loader: Combined DataLoader for all test sets
        If return_individual=True:
            dict: {
                'combined': combined DataLoader,
                'individual': {csv_path: DataLoader, ...}
            }

    CSV Format:
        - Required columns: 'image_path', 'fraud_type'
        - fraud_type values: 'genuine' or 'tamper'
    """
    # Handle single CSV path
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    # Create cache key based on CSV paths
    cache_key = hashlib.md5('|'.join(sorted(csv_paths)).encode()).hexdigest()
    cache_dir = os.path.join(os.path.dirname(csv_paths[0]) if csv_paths else '.', '.test_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'test_data_{cache_key}.pkl')

    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        try:
            print(f"\n{'='*70}")
            print("Loading Test Dataset from Cache")
            print(f"{'='*70}")
            print(f"Cache file: {cache_file}")

            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            all_image_paths = cache_data['all_image_paths']
            all_labels = cache_data['all_labels']
            csv_data = cache_data['csv_data']

            print(f"✅ Loaded {len(all_image_paths)} images from cache")
            print(f"{'='*70}\n")

            # Skip to dataset creation
            class_names = ['genuine', 'tamper']

        except Exception as e:
            print(f"⚠️  Cache loading failed ({e}), rebuilding dataset...")
            use_cache = False  # Force rebuild

    if not use_cache or not os.path.exists(cache_file):
        print(f"\n{'='*70}")
        print("Loading Test Dataset from CSV")
        print(f"{'='*70}")
        print(f"Number of CSV files: {len(csv_paths)}")

        # Class mapping
        class_names = ['genuine', 'tamper']
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Store data per CSV for individual loaders
        csv_data = {}

        # Load all CSV files
        all_image_paths = []
        all_labels = []

        for csv_path in csv_paths:
            print(f"\nLoading: {csv_path}")

            if not os.path.exists(csv_path):
                logging.warning(f"CSV file not found: {csv_path}")
                continue

            try:
                # Read CSV with explicit dtypes and prevent NaN conversion
                # keep_default_na=False prevents empty strings from becoming NaN (float)
                df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

                # Validate columns
                if 'image_path' not in df.columns:
                    logging.error(f"Missing 'image_path' column in {csv_path}")
                    continue

                if 'fraud_type' not in df.columns:
                    logging.error(f"Missing 'fraud_type' column in {csv_path}")
                    continue

                # Load data for this CSV
                csv_image_paths = []
                csv_labels = []

                valid_count = 0
                for _, row in df.iterrows():
                    img_path = row['image_path']
                    fraud_type = str(row['fraud_type']).strip().lower()

                    # Map fraud_type to class index
                    if fraud_type not in class_to_idx:
                        logging.warning(f"Unknown fraud_type '{fraud_type}' in {csv_path}, skipping")
                        continue

                    # Check if image exists
                    if not os.path.exists(img_path):
                        logging.warning(f"Image not found: {img_path}")
                        continue

                    # Validate image
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        # Re-open after verify
                        with Image.open(img_path) as img:
                            img.load()

                        # Add to combined dataset
                        all_image_paths.append(img_path)
                        all_labels.append(class_to_idx[fraud_type])

                        # Add to individual dataset
                        csv_image_paths.append(img_path)
                        csv_labels.append(class_to_idx[fraud_type])

                        valid_count += 1

                    except Exception as e:
                        logging.warning(f"Corrupted image {img_path}: {e}")

                print(f"  Loaded {valid_count} valid images")

                # Store data for this CSV
                if valid_count > 0:
                    csv_data[csv_path] = {
                        'image_paths': csv_image_paths,
                        'labels': csv_labels
                    }

            except Exception as e:
                logging.error(f"Error reading {csv_path}: {e}")

        # Save to cache
        try:
            cache_data = {
                'all_image_paths': all_image_paths,
                'all_labels': all_labels,
                'csv_data': csv_data
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"\n💾 Saved test dataset cache to {cache_file}")
        except Exception as e:
            print(f"\n⚠️  Failed to save cache: {e}")

    if len(all_image_paths) == 0:
        raise ValueError("No valid images found in any CSV file!")

    # Print class distribution
    from collections import Counter
    class_counts = Counter(all_labels)

    print(f"\n{'='*70}")
    print("Test Dataset Summaryyyyy")
    print(f"{'='*70}")
    print(f"Total imagessssssss: {len(all_image_paths)}")
    print("\nClass Distribution:")
    print("-" * 30)
    for idx, class_name in enumerate(class_names):
        count = class_counts.get(idx, 0)
        percentage = (count / len(all_labels)) * 100 if all_labels else 0
        print(f"{class_name}: {count} images ({percentage:.1f}%)")
    print(f"{'='*70}\n")

    # Create test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Direct resize to target size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create combined dataset
    combined_dataset = TestDatasetFromCSV(
        all_image_paths,
        all_labels,
        transform=test_transform,
        class_names=class_names
    )

    # Create combined dataloader
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Combined Test DataLoader created: {len(combined_loader)} batches\n")

    # If individual loaders requested, create them
    if return_individual:
        individual_loaders = {}

        print(f"{'='*70}")
        print("Creating Individual DataLoaders per CSV")
        print(f"{'='*70}")

        for csv_path, data in csv_data.items():
            dataset = TestDatasetFromCSV(
                data['image_paths'],
                data['labels'],
                transform=test_transform,
                class_names=class_names
            )

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            individual_loaders[csv_path] = loader
            csv_name = os.path.basename(csv_path)
            print(f"  {csv_name}: {len(loader)} batches, {len(dataset)} images")

        print(f"{'='*70}\n")

        return {
            'combined': combined_loader,
            'individual': individual_loaders,
            'csv_names': {path: os.path.basename(path) for path in csv_data.keys()}
        }
    else:
        return combined_loader
