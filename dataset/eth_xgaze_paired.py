"""
ETH-XGaze Paired Frame Dataset for FAGE Eye-Only Training

Adapts ETHXGazeLMDBDataset to return paired frames compatible with GazeCapture format.
"""
import torch
import numpy as np
import cv2
import random
import logging
from PIL import Image
from torchvision import transforms
from .eth_xgaze import ETHXGazeLMDBDataset

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class ETHXGazePairedDataset(ETHXGazeLMDBDataset):
    """
    ETH-XGaze dataset with paired frame support for FAGE training.

    Returns the same format as HDFDataset (GazeCapture):
    - source frame: provides eye appearance
    - target frame: provides gaze direction + GT
    - dual-scale eye crops: large (input) + tight (supervision)
    """

    def __init__(self, lmdb_path,
                 transform=None,
                 resolution=256,
                 aug=False,
                 split="train",
                 split_ratio=(0.8, 0.1, 0.1),
                 seed=41,
                 frame_offset_range=2,
                 eye_crop_size=(64, 64),
                 eye_expand_ratio=1.5,
                 input_eye_crop_size=None,
                 input_eye_expand_ratio=None):
        """
        Args:
            lmdb_path: LMDB database path
            resolution: Image resolution
            aug: Enable data augmentation
            split: 'train', 'val', or 'test'
            split_ratio: Train/val/test split ratio
            seed: Random seed
            frame_offset_range: Frame offset range ±N for pairing
            eye_crop_size: Tight eye crop size (H, W) for supervision
            eye_expand_ratio: Expansion ratio for tight crop
            input_eye_crop_size: Large eye crop size (H, W) for model input
            input_eye_expand_ratio: Expansion ratio for large crop
        """
        # Initialize parent class (without return_reference to avoid overhead)
        super().__init__(
            lmdb_path=lmdb_path,
            transform=None,  # We'll handle transform ourselves
            resolution=resolution,
            aug=aug,
            split=split,
            split_ratio=split_ratio,
            seed=seed,
            eye_scale_factor=1.0,  # We'll handle scaling in _crop_eyes
            use_precomputed_masks=True,
            convert_to_square=False,
            return_reference=False,  # Don't need reference for paired training
        )

        self.frame_offset_range = frame_offset_range
        self.eye_crop_size = tuple(eye_crop_size)
        self.eye_expand_ratio = eye_expand_ratio
        self.input_eye_crop_size = tuple(input_eye_crop_size) if input_eye_crop_size else self.eye_crop_size
        self.input_eye_expand_ratio = input_eye_expand_ratio if input_eye_expand_ratio else eye_expand_ratio

        # Setup transform
        if transform is None:
            self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.to_tensor = transform

        logging.info(f"ETH-XGaze Paired Dataset [{self.split}]: {len(self)} frames, "
                     f"{len(self._subject_to_indices)} subjects")
        logging.info(f"  Eye crops: tight={self.eye_crop_size} (ratio={self.eye_expand_ratio}), "
                     f"input={self.input_eye_crop_size} (ratio={self.input_eye_expand_ratio})")

    def _crop_eyes_from_mask(self, image_np, mask_np, crop_size, expand_ratio):
        """
        Crop left and right eyes from image using mask.

        Args:
            image_np: [H, W, 3] uint8
            mask_np: [H, W] uint8 (binary mask, 255=eye region)
            crop_size: (crop_h, crop_w)
            expand_ratio: Expansion ratio for eye bbox

        Returns:
            crops: list of 2 numpy arrays [crop_h, crop_w, 3]
            bbox_list: list of 8 floats (normalized coordinates)
        """
        H, W = image_np.shape[:2]
        crop_h, crop_w = crop_size

        # Binarize mask
        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Collect eye regions
        eye_info = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter small regions
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                eye_info.append({
                    'center': (int(cx), int(cy)),
                    'radius': radius,
                    'area': area
                })

        # Sort by area (largest first), take top 2 (left + right eyes)
        eye_info = sorted(eye_info, key=lambda x: x['area'], reverse=True)[:2]
        # Sort by x-coordinate (left to right)
        eye_info = sorted(eye_info, key=lambda x: x['center'][0])

        crops = []
        bbox_list = []

        for eye in eye_info:
            cx, cy = eye['center']
            radius = eye['radius']

            # Calculate bbox with expansion
            r = int(radius * expand_ratio)
            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(W, cx + r)
            y2 = min(H, cy + r)

            # Crop and resize
            crop = image_np[y1:y2, x1:x2]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
                crops.append(crop_resized)
                bbox_list.extend([x1/W, y1/H, x2/W, y2/H])
            else:
                crops.append(np.zeros((crop_h, crop_w, 3), dtype=np.uint8))
                bbox_list.extend([0, 0, 0, 0])

        # Ensure we have exactly 2 eyes (pad with zeros if needed)
        while len(crops) < 2:
            crops.append(np.zeros((crop_h, crop_w, 3), dtype=np.uint8))
            bbox_list.extend([0, 0, 0, 0])

        return crops[:2], bbox_list[:8]

    def _crop_eyes(self, image_np, mask_np):
        """
        Crop eyes with dual scales: large (input) + tight (supervision).

        Args:
            image_np: [H, W, 3] uint8
            mask_np: [H, W] uint8

        Returns:
            input_tensor: [3, in_h, in_w*2] - large crop for model input (width concat)
            target_tensor: [3, crop_h, crop_w*2] - tight crop for supervision (width concat)
            bbox_tensor: [8] - normalized bbox coordinates (based on tight crop)
        """
        # Large crops (model input)
        input_crops, _ = self._crop_eyes_from_mask(
            image_np, mask_np, self.input_eye_crop_size, self.input_eye_expand_ratio
        )

        # Tight crops (supervision)
        target_crops, bbox_list = self._crop_eyes_from_mask(
            image_np, mask_np, self.eye_crop_size, self.eye_expand_ratio
        )

        # Convert to tensors and width-concat left + right eyes
        input_tensor = torch.cat([
            self.to_tensor(Image.fromarray(input_crops[0])),
            self.to_tensor(Image.fromarray(input_crops[1]))
        ], dim=2)  # [3, in_h, in_w*2]

        target_tensor = torch.cat([
            self.to_tensor(Image.fromarray(target_crops[0])),
            self.to_tensor(Image.fromarray(target_crops[1]))
        ], dim=0)  # [6, crop_h, crop_w]

        bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)

        return input_tensor, target_tensor, bbox_tensor

    def _get_paired_frame_idx(self, current_idx, subject_id):
        """
        Get a paired frame index from the same subject.

        Args:
            current_idx: Current data index
            subject_id: Subject ID

        Returns:
            paired_idx: Paired frame index
        """
        # Get all indices for this subject
        subject_indices = self._subject_to_indices.get(subject_id, [])

        if len(subject_indices) < 2:
            # Not enough frames, return same index
            return current_idx

        # Try to find a frame within offset range
        valid_indices = [idx for idx in subject_indices if idx != current_idx]

        if not valid_indices:
            return current_idx

        # Randomly select from valid indices
        return random.choice(valid_indices)

    def __getitem__(self, idx):
        """
        Returns paired frames (source + target) compatible with GazeCapture format.

        Returns:
            dict with:
                source_image: [3, H, W] - full face image
                source_gaze: [2] - gaze angles (pitch, yaw) in radians
                source_head: [2] - head pose angles (pitch, yaw) in radians
                source_input_eye_crops: [6, in_h, in_w] - large eye crops for model input
                source_eye_crops: [6, h, w] - tight eye crops (backup GT)
                source_eye_bbox: [8] - normalized bbox coordinates
                target_image: [3, H, W] - full face image
                target_gaze: [2] - gaze angles
                target_head: [2] - head pose angles
                target_input_eye_crops: [6, in_h, in_w] - large eye crops (unused)
                target_eye_crops: [6, h, w] - tight eye crops for supervision
                target_eye_bbox: [8] - normalized bbox coordinates
                subject_idx: int - subject index
        """
        # Get source frame data using parent class method
        data_idx = self.indices[idx]

        # Use parent's __getitem__ to get source frame
        source_data = super().__getitem__(idx)

        source_image = source_data['image']  # [3, H, W]
        source_mask = source_data['mask']    # [H, W]
        source_gaze = source_data['norm_gaze']  # [2]
        source_head = source_data['norm_head']  # [2]
        subject_id = source_data['subject_id']

        # Get paired target frame
        target_idx_global = self._get_paired_frame_idx(data_idx, subject_id)
        target_idx_local = np.where(self.indices == target_idx_global)[0]

        if len(target_idx_local) > 0:
            target_data = super().__getitem__(target_idx_local[0])
        else:
            # Fallback: use source as target
            target_data = source_data

        target_image = target_data['image']
        target_mask = target_data['mask']
        target_gaze = target_data['norm_gaze']
        target_head = target_data['norm_head']

        # Convert tensors back to numpy for eye cropping
        # Source image: [3, H, W] -> [H, W, 3]
        source_np = ((source_image.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        source_mask_np = (source_mask.numpy() * 255).astype(np.uint8)

        # Target image: [3, H, W] -> [H, W, 3]
        target_np = ((target_image.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        target_mask_np = (target_mask.numpy() * 255).astype(np.uint8)

        # Crop eyes with dual scales
        src_input, src_target, source_eye_bbox = self._crop_eyes(source_np, source_mask_np)
        tgt_input, tgt_target, target_eye_bbox = self._crop_eyes(target_np, target_mask_np)

        # Re-apply transform to full images (they were already transformed by parent)
        # Just use them directly

        # Get subject index
        subject_idx = list(self._subject_to_indices.keys()).index(subject_id) if subject_id in self._subject_to_indices else 0

        return {
            'source_image': source_image,
            'source_gaze': source_gaze,
            'source_head': source_head,
            'source_input_eye_crops': src_input,
            'source_eye_crops': src_target,
            'source_eye_bbox': source_eye_bbox,

            'target_image': target_image,
            'target_gaze': target_gaze,
            'target_head': target_head,
            'target_input_eye_crops': tgt_input,
            'target_eye_crops': tgt_target,
            'target_eye_bbox': target_eye_bbox,

            'subject_idx': subject_idx,
        }

    @property
    def num_subjects(self):
        return len(self._subject_to_indices)


if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if len(sys.argv) > 1:
        lmdb_path = sys.argv[1]
    else:
        lmdb_path = "/mnt/data/xhy/GC-GAN/ETH-GAZE/Vae-Gaze/dataset_lmdb"

    print("=" * 60)
    print("Testing ETH-XGaze Paired Dataset")
    print("=" * 60)

    dataset = ETHXGazePairedDataset(
        lmdb_path=lmdb_path,
        resolution=256,
        aug=False,
        split="train",
        frame_offset_range=2,
        eye_crop_size=(64, 64),
        eye_expand_ratio=1.5,
        input_eye_crop_size=(80, 80),
        input_eye_expand_ratio=2.0,
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    batch = next(iter(dataloader))

    print("✅ Dataset test successful!")
    print(f"Source image: {batch['source_image'].shape}")
    print(f"Source input eye crops: {batch['source_input_eye_crops'].shape}")
    print(f"Source eye crops: {batch['source_eye_crops'].shape}")
    print(f"Target eye crops: {batch['target_eye_crops'].shape}")
    print(f"Source gaze: {batch['source_gaze'].shape}")
    print(f"Target gaze: {batch['target_gaze'].shape}")
    print(f"Subject idx: {batch['subject_idx']}")
