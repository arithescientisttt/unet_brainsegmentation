import os
import random
import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from utils import crop_image, pad_image, resize_image, normalize_intensity


class BrainSegmentationDataset(Dataset):
    """Dataset class for brain MRI segmentation using FLAIR images."""

    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=True,
        validation_cases=10,
        seed=42,
    ):
        assert subset in ["all", "train", "validation"], "Invalid subset specified."

        # Initialize volume and mask dictionaries
        volumes = {}
        masks = {}

        # Read and store images
        print(f"Loading {subset} images...")
        for dirpath, dirnames, filenames in os.walk(images_dir):
            img_slices, mask_slices = [], []
            for file in sorted([f for f in filenames if f.endswith(".tif")],
                               key=lambda x: int(x.split(".")[-2].split("_")[4])):
                filepath = os.path.join(dirpath, file)
                if "mask" in file:
                    mask_slices.append(imread(filepath, as_gray=True))
                else:
                    img_slices.append(imread(filepath))

            if img_slices:
                patient_id = os.path.basename(dirpath)
                volumes[patient_id] = np.array(img_slices[1:-1])  # Exclude first/last slices
                masks[patient_id] = np.array(mask_slices[1:-1])

        # Sort patients and split for validation if necessary
        self.patients = sorted(volumes.keys())
        if subset != "all":
            random.seed(seed)
            validation_patients = random.sample(self.patients, k=validation_cases)
            self.patients = (
                validation_patients if subset == "validation"
                else sorted(set(self.patients) - set(validation_patients))
            )

        # Preprocess volumes and masks
        print(f"Preprocessing {subset} volumes...")
        self.volumes = [(volumes[p], masks[p]) for p in self.patients]
        self.volumes = [crop_sample(v) for v in self.volumes]  # Crop
        self.volumes = [pad_sample(v) for v in self.volumes]  # Pad to square
        self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]  # Resize
        self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]  # Normalize

        # Calculate slice weights for random sampling
        print(f"Setting up slice weights for {subset} dataset...")
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for _, m in self.volumes]
        self.slice_weights = [
            (weights + (weights.sum() * 0.1 / len(weights))) / (weights.sum() * 1.1)
            for weights in self.slice_weights
        ]

        # Ensure mask has an additional channel dimension
        self.volumes = [(v, m[..., np.newaxis]) for v, m in self.volumes]

        # Create a global index for patient slices
        print(f"Dataset ready for {subset}.")
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = [
            (patient_idx, slice_idx)
            for patient_idx, slice_count in enumerate(num_slices)
            for slice_idx in range(slice_count)
        ]

        self.random_sampling = random_sampling
        self.transform = transform

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        # Get patient and slice index
        patient_idx, slice_idx = self.patient_slice_index[idx]

        # Perform random sampling if enabled
        if self.random_sampling:
            patient_idx = np.random.randint(len(self.volumes))
            slice_idx = np.random.choice(
                range(self.volumes[patient_idx][0].shape[0]),
                p=self.slice_weights[patient_idx],
            )

        volume, mask = self.volumes[patient_idx]
        image = volume[slice_idx]
        mask = mask[slice_idx]

        # Apply any transformations (i
