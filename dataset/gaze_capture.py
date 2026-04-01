import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import h5py
import random
import logging
import json
from PIL import Image
from torchvision import transforms
# from io import BytesIO
from PIL import Image


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

class HDFDataset(Dataset):
    """Dataset for loading eye gaze data from HDF5 files, with train/val/test split support"""

    def __init__(self, hdf_file_path,
                 prefixes=None,
                 is_bgr=False,
                 get_2nd_sample=False,
                 pick_at_least_per_person=None,
                 num_labeled_samples=None,
                 sample_target_label=False,
                 split="train",
                 split_ratio=(0.8, 0.1, 0.1),
                 seed=42,
                 aug=False,
                 transform=None,
                 split_json_path="/home/xuhy/PycharmProjects/Unet-Gaze/gazecapture_split.json",
                 reference_search_radius=20,
                 return_eye_reference=False,
                 return_original_eye_reference=False):
        """
        Initialize the dataset.
        Args:
            hdf_file_path: Path to HDF5 file
            prefixes: List of prefixes to use (default: None, use all)
            is_bgr: Whether images are in BGR format (default: False)
            get_2nd_sample: Whether to get second sample (default: False)
            pick_at_least_per_person: Minimum number of samples per person (default: None)
            num_labeled_samples: Number of labeled samples to use (default: None)
            sample_target_label: Whether to sample target labels (default: False)
            split: 'train', 'val', or 'test' (default: 'train')
            split_ratio: tuple, e.g. (0.8,0.1,0.1) for train/val/test (only used if split_json_path is None)
            seed: random seed for reproducibility (only used if split_json_path is None)
            aug: Whether to apply data augmentation (default: False)
                 - Random horizontal flip (50%)
                 - Random color jitter (brightness, contrast, saturation, hue) (80%)
                 - Random rotation (±5 degrees) (50%)
            split_json_path: Path to JSON file with predefined train/val/test splits
                           - Default: /home/xuhy/PycharmProjects/Unet-Gaze/gazecapture_split.json
                           - explicit path: use the specified JSON file
                           - "auto": automatically search for gazecapture_split.json in current dir and parent dirs
                           - None: use random splitting based on split_ratio
            reference_search_radius: Search radius for reference image selection (default: 20)
                           - Only used when return_eye_reference=True
                           - Limits search to nearby frames within the same sequence/session
                           - Ensures background consistency by temporal proximity
                           - Set to None or large value (e.g., 10000) to search all frames of the subject
            return_eye_reference: Whether to return eye texture region from REFERENCE (default: False)
                           - Extracts left and right eye regions from forward-looking reference
                           - Returns 256×512 tensor: [left eye 256×256 | right eye 256×256]
                           - Automatically loads reference image (no need to set return_reference)
            return_original_eye_reference: Whether to return eye texture region from ORIGINAL IMAGE (default: False)
                           - Extracts left and right eye regions from current sample image
                           - Returns 256×512 tensor: [left eye 256×256 | right eye 256×256]
                           - Does not require reference image
        """
        assert os.path.isfile(hdf_file_path), f"HDF5 file not found: {hdf_file_path}"
        self.get_2nd_sample = get_2nd_sample
        self.hdf_path = hdf_file_path
        self.hdf = None
        self.is_bgr = is_bgr
        self.sample_target_label = sample_target_label
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.aug = aug
        self.reference_search_radius = reference_search_radius
        self.return_eye_reference = return_eye_reference
        self.return_original_eye_reference = return_original_eye_reference

        # Validate: 两种模式互斥
        if self.return_original_eye_reference and self.return_eye_reference:
            raise ValueError("return_original_eye_reference and return_eye_reference are mutually exclusive. Choose one.")

        if self.return_eye_reference:
            logging.info("dataset return eye_texture")
        if self.return_original_eye_reference:
            logging.info("dataset return orginal_eye_texture")
        # Handle split_json_path parameter
        if split_json_path is not None:
            # Use the provided path (could be default or user-specified)
            if os.path.isfile(split_json_path):
                self.split_json_path = split_json_path
                logging.info(f"Using split JSON file: {self.split_json_path}")
            else:
                logging.warning(f"Split JSON file not found: {split_json_path}, will use random split")
                self.split_json_path = None
        else:
            # split_json_path is None, use random split
            self.split_json_path = None

        # Initialize image transforms
        if transform is None:
            self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.to_tensor = transform

        # Load dataset information
        with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as h5f:
            hdf_keys = sorted(list(h5f.keys()))
            # print("hdf_keys", hdf_keys)
            if prefixes is None:
                self.prefixes = hdf_keys
            else:
                self.prefixes = [k for k in prefixes if k in h5f]

            # Filter by minimum samples per person
            if pick_at_least_per_person is not None:
                self.prefixes = [k for k in self.prefixes
                               if k in h5f and len(next(iter(h5f[k].values()))) >= pick_at_least_per_person]

            # --- Split by prefix ---
            # Use JSON split if provided, otherwise use random split
            if self.split_json_path is not None:
                # Load split from JSON file
                assert os.path.isfile(self.split_json_path), f"Split JSON file not found: {self.split_json_path}"
                with open(self.split_json_path, 'r') as f:
                    all_splits = json.load(f)
                
                # Get the split prefixes from JSON
                if self.split not in all_splits:
                    raise ValueError(f"Split '{self.split}' not found in JSON file. Available splits: {list(all_splits.keys())}")
                
                json_prefixes = all_splits[self.split]
                # Filter to only include prefixes that exist in both JSON and HDF5
                split_prefixes = [k for k in json_prefixes if k in self.prefixes]
                
                logging.info(f"Using JSON split '{self.split}' from {self.split_json_path}")
                logging.info(f"JSON defines {len(json_prefixes)} prefixes, {len(split_prefixes)} exist in HDF5")
                
                self.prefixes = split_prefixes
            else:
                # Use random split based on split_ratio
                rng = np.random.RandomState(self.seed)
                num_prefixes = len(self.prefixes)
                idxs = np.arange(num_prefixes)
                rng.shuffle(idxs)
                n_train = int(self.split_ratio[0] * num_prefixes)
                n_val = int(self.split_ratio[1] * num_prefixes)
                if self.split == "train":
                    split_prefixes = [self.prefixes[i] for i in idxs[:n_train]]
                elif self.split == "val":
                    split_prefixes = [self.prefixes[i] for i in idxs[n_train:n_train+n_val]]
                elif self.split == "test":
                    split_prefixes = [self.prefixes[i] for i in idxs[n_train+n_val:]]
                else:
                    raise ValueError(f"Unknown split: {self.split}")
                self.prefixes = split_prefixes

            # Create index to query mapping
            self.index_to_query = sum([[(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                                     for prefix in self.prefixes], [])

            # Build prefix_to_indices mapping for reference image selection
            # 只有在需要参考图时才构建 (return_eye_reference=True)
            if self.return_eye_reference:
                self.prefix_to_indices = {}
                for idx, (prefix, _) in enumerate(self.index_to_query):
                    if prefix not in self.prefix_to_indices:
                        self.prefix_to_indices[prefix] = []
                    self.prefix_to_indices[prefix].append(idx)
                logging.info(f"📊 Built prefix_to_indices mapping: {len(self.prefix_to_indices)} subjects")

                # Filter out subjects with only 1 sample (cannot find reference)
                single_sample_subjects = [prefix for prefix, indices in self.prefix_to_indices.items() if len(indices) == 1]
                if len(single_sample_subjects) > 0:
                    logging.warning(f"⚠️  Filtering {len(single_sample_subjects)} subjects with only 1 sample")
                    # Remove indices of single-sample subjects from index_to_query
                    indices_to_remove = set()
                    for prefix in single_sample_subjects:
                        indices_to_remove.update(self.prefix_to_indices[prefix])
                        del self.prefix_to_indices[prefix]

                    # Rebuild index_to_query without these samples
                    self.index_to_query = [item for idx, item in enumerate(self.index_to_query) if idx not in indices_to_remove]

                    # Rebuild prefix_to_indices with new indices
                    self.prefix_to_indices = {}
                    for idx, (prefix, _) in enumerate(self.index_to_query):
                        if prefix not in self.prefix_to_indices:
                            self.prefix_to_indices[prefix] = []
                        self.prefix_to_indices[prefix].append(idx)

                    logging.info(f"✅ After filtering: {len(self.index_to_query)} samples, {len(self.prefix_to_indices)} subjects")

            # Sample subset if specified
            if num_labeled_samples is not None:
                ra = list(range(len(self.index_to_query)))
                random.seed(0)
                random.shuffle(ra)
                ra = ra[:num_labeled_samples]
                list.sort(ra)
                self.index_to_query = [self.index_to_query[i] for i in ra]

    def _apply_augmentation(self, image, mask, norm_gaze, norm_head):
        """
        Apply data augmentation to image, mask, and gaze/head pose information

        Args:
            image: PIL Image - input image
            mask: numpy array - eye mask
            norm_gaze: numpy array [2] - gaze direction (pitch, yaw)
            norm_head: numpy array [2] - head pose (pitch, yaw)

        Returns:
            image_aug: PIL Image - augmented image
            mask_aug: numpy array - augmented mask
            norm_gaze_aug: numpy array [2] - adjusted gaze direction
            norm_head_aug: numpy array [2] - adjusted head pose
        """
        # Make copies to avoid modifying originals and ensure contiguous memory
        mask = np.ascontiguousarray(mask)  # Ensure mask is a contiguous copy
        norm_gaze_aug = norm_gaze.copy()
        norm_head_aug = norm_head.copy()

        # 1. Random horizontal flip (50% probability)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.fliplr(mask).copy()  # .copy() to avoid negative stride issue
            # Horizontal flip reverses yaw angle (left-right direction)
            norm_gaze_aug[1] = -norm_gaze[1]  # yaw
            norm_head_aug[1] = -norm_head[1]  # yaw

        # 2. Random color jitter (80% probability)
        if random.random() < 0.8:
            # Brightness jitter ±20%
            brightness_factor = random.uniform(0.8, 1.2)
            image = transforms.functional.adjust_brightness(image, brightness_factor)

            # Contrast jitter ±20%
            contrast_factor = random.uniform(0.8, 1.2)
            image = transforms.functional.adjust_contrast(image, contrast_factor)

            # Saturation jitter ±20%
            saturation_factor = random.uniform(0.8, 1.2)
            image = transforms.functional.adjust_saturation(image, saturation_factor)

            # Hue jitter ±10 degrees
            hue_factor = random.uniform(-0.05, 0.05)
            image = transforms.functional.adjust_hue(image, hue_factor)

        # 3. Random rotation (±5 degrees, 50% probability)
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
            # Rotate mask (use nearest neighbor interpolation to preserve binary mask)
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST, fillcolor=0)
            # Ensure contiguous array after rotation
            mask = np.ascontiguousarray(np.array(mask_pil), dtype=np.float32) / 255.0

            # For small rotation angles, the effect on gaze is negligible
            # If precise adjustment is needed, apply rotation matrix

        return image, mask, norm_gaze_aug, norm_head_aug

    def _extract_eye_texture_from_mask(self, ref_image, ref_mask, eye_size=256):
        """
        从参考图片中提取左右眼纹理区域，直接根据mask位置裁剪

        GazeCapture的mask已经标注了眼睛位置，直接找到mask区域并裁剪对应的pixel区域即可
        数据集内掩码是精确解析掩码

        Args:
            ref_image: PIL.Image - 参考图像 (256×256)
            ref_mask: numpy.ndarray - 眼部掩码 (256×256)
            eye_size: int - 每只眼睛的正方形尺寸 (默认256)

        Returns:
            eye_texture: PIL.Image - 左右眼拼接纹理 (256×512)
                         左眼在左半部分(0:256)，右眼在右半部分(256:512)
            eye_centers: tuple of (left_center, right_center)
                         每个 center 是 (cx, cy) 归一化到 [0, 1]
                         如果某只眼睛缺失，对应 center 为 None
        """
        # 转换为numpy array
        if isinstance(ref_image, Image.Image):
            ref_image_array = np.array(ref_image)
        else:
            ref_image_array = ref_image

        # 确保mask是numpy array
        if isinstance(ref_mask, torch.Tensor):
            mask_array = ref_mask.cpu().numpy()
        else:
            mask_array = ref_mask

        # 归一化mask到0-1
        if mask_array.max() > 1.0:
            mask_array = mask_array / 255.0

        h, w = mask_array.shape[:2]
        mid_x = w // 2

        # 分别处理左右两半
        eye_regions = []
        eye_regions_bbox = []  # 存储 bbox: [left_x1, left_y1, left_x2, left_y2, right_x1, right_y1, right_x2, right_y2]

        # 左半部分（左眼）
        left_mask = mask_array[:, :mid_x]
        if left_mask.max() > 0.1:  # 有眼睛区域
            rows, cols = np.where(left_mask > 0.1)
            if len(rows) > 0:
                y1, y2 = rows.min(), rows.max() + 1
                x1, x2 = cols.min(), cols.max() + 1
                # 存储 bbox（归一化到 [0, 1]）
                eye_regions_bbox.extend([x1 / w, y1 / h, x2 / w, y2 / h])
                eye_regions.append(('left', x1, y1, x2, y2))
        # else:
        #     # 左眼缺失，使用默认 bbox
        #     eye_regions_bbox.extend([0.1, 0.3, 0.4, 0.7])

        # 右半部分（右眼）
        right_mask = mask_array[:, mid_x:]
        if right_mask.max() > 0.1:  # 有眼睛区域
            rows, cols = np.where(right_mask > 0.1)
            if len(rows) > 0:
                y1, y2 = rows.min(), rows.max() + 1
                x1, x2 = cols.min() + mid_x, cols.max() + mid_x + 1  # 加上偏移
                # 存储 bbox（归一化到 [0, 1]）
                eye_regions_bbox.extend([x1 / w, y1 / h, x2 / w, y2 / h])
                eye_regions.append(('right', x1, y1, x2, y2))
        # else:
        #     # 右眼缺失，使用默认 bbox
        #     eye_regions_bbox.extend([0.6, 0.3, 0.9, 0.7])

        # 创建输出画布 (256×512)
        output_canvas = np.zeros((eye_size, eye_size * 2, 3), dtype=np.uint8)

        if len(eye_regions) == 0:
            # 回退：返回整张脸的左右复制
            face_resized = cv2.resize(ref_image_array, (eye_size, eye_size), interpolation=cv2.INTER_LANCZOS4)
            output_canvas[:, 0:eye_size, :] = face_resized
            output_canvas[:, eye_size:eye_size*2, :] = face_resized
        else:
            # 处理每只眼睛
            for idx, (side, x1, y1, x2, y2) in enumerate(eye_regions):
                # 裁剪眼睛区域
                eye_crop = ref_image_array[y1:y2, x1:x2].copy()

                # Resize到256×256
                if eye_crop.size > 0:
                    eye_crop_resized = cv2.resize(eye_crop, (eye_size, eye_size), interpolation=cv2.INTER_LANCZOS4)

                    # 放置到对应位置
                    if side == 'left':
                        output_canvas[:, 0:eye_size, :] = eye_crop_resized
                    else:  # right
                        output_canvas[:, eye_size:eye_size*2, :] = eye_crop_resized

            # # 如果只有一只眼睛，镜像复制
            # if len(eye_regions) == 1:
            #     if eye_regions[0][0] == 'left':
            #         output_canvas[:, eye_size:eye_size*2, :] = output_canvas[:, 0:eye_size, :]
            #     else:
            #         output_canvas[:, 0:eye_size, :] = output_canvas[:, eye_size:eye_size*2, :]

        eye_texture = Image.fromarray(output_canvas)

        # 返回眼睛纹理和 bbox
        return eye_texture, eye_regions_bbox

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.index_to_query)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Args:
            idx: Index of the sample
        Returns:
            dict: Sample dictionary containing:
                - image: Original face image [3, H, W]
                - norm_gaze: Normalized gaze direction [2]
                - norm_head: Normalized head pose [2]
                - eyes_mask: Combined eyes mask [H, W]
                - left_eye_mask: Left eye mask [H, W]
                - right_eye_mask: Right eye mask [H, W]
                - boundary_mask: Eye boundary mask [H, W]
        """
        # Lazy loading of HDF5 file
        if self.hdf is None:
            self.hdf = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True)

        # Get sample data
        key, index = self.index_to_query[idx]
        group = self.hdf[key]
        # print(f"group type is {type(group)}")
        # print(f"group is {group}")


        # Get original face image and labels
        image = group['pixels'][index, ...]  # [H, W, 3], uint8 or float32
        norm_gaze = group['labels'][index, 0:2]  # [2], float32
        norm_head = group['labels'][index, 2:4]  # [2], float32
     
        if 'face_parsing' in group:
            # face_parsing contains labels: 0=background, 1=left_eye, 2=right_eye, 3=left_iris, 4=right_iris
            face_parsing = group['face_parsing'][index, ...]  # [H, W], uint8

            # Create mask using minimum enclosing circle + rectangle
            mask = np.zeros(face_parsing.shape[:2], dtype=np.float32)

            # Process left eye (label=1) and right eye (label=2)
            for eye_label in [1, 2]:  # 1=left_eye, 2=right_eye
                eye_mask = (face_parsing == eye_label).astype(np.uint8)

                if eye_mask.sum() > 0:  # Check if eye region exists
                    # Find all non-zero points
                    points = cv2.findNonZero(eye_mask)

                    if points is not None and len(points) >= 5:  # Need at least 5 points for circle
                        # Get minimum enclosing circle
                        (cx, cy), radius = cv2.minEnclosingCircle(points)
                        cx, cy = int(cx), int(cy)
                        radius = int(1.4 * radius)  # Scale radius by 1.2

                        # Draw filled rectangle centered at (cx, cy) with side length 2*radius
                        cv2.rectangle(mask,
                                   (cx - radius, cy - radius),
                                   (cx + radius, cy + radius),
                                   1, -1)
        elif 'mask' in group:
            # Use precomputed mask if available (legacy support)
            mask = group["mask"][index,...]

        # Convert image to PIL Image BEFORE augmentation
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                # Convert to uint8 if it's float
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            # Convert to PIL Image (expects HWC format)
            if image.ndim == 3 and image.shape[2] == 3:
                image = Image.fromarray(image)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")

        # Apply data augmentation if enabled (BEFORE tensor conversion)
        # if self.aug:
        #     # norm_gaze and norm_head are still numpy arrays at this point
        #     image, mask, norm_gaze, norm_head = self._apply_augmentation(
        #         image, mask, norm_gaze, norm_head
        #     )

        # Convert gaze/head to torch tensors AFTER augmentation
        # Use ascontiguousarray to ensure contiguous memory layout
        norm_gaze = torch.from_numpy(np.ascontiguousarray(norm_gaze, dtype=np.float32))
        norm_head = torch.from_numpy(np.ascontiguousarray(norm_head, dtype=np.float32))

        # Apply transforms: ToTensor() + Normalize()
        # ToTensor(): [0, 255] -> [0, 1] and HWC -> CHW
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)): [0, 1] -> [-1, 1]
        image_tensor = self.to_tensor(image)

        # Extract eye texture from ORIGINAL image if requested
        if self.return_original_eye_reference:
            # Extract left and right eye regions from current image
            original_eye_texture_pil, eye_bbox = self._extract_eye_texture_from_mask(
                image, mask, eye_size=256
            )
            # Transform to tensor (256×512 -> [3, 256, 512])
            original_eye_texture_tensor = self.to_tensor(original_eye_texture_pil)

            # 转换 bbox 为 tensor [8]
            eye_bbox_tensor = torch.tensor(eye_bbox, dtype=torch.float32)

        # Build sample dictionary
        sample = {
            'image': image_tensor,         # Original face image [3, H, W]
            'norm_gaze': norm_gaze,        # Normalized gaze direction [2]
            'norm_head': norm_head,        # Normalized head pose [2]
            "mask": torch.from_numpy(np.ascontiguousarray(mask))  # Ensure contiguous memory and convert to tensor
        }

        # Add original eye texture if requested
        if self.return_original_eye_reference:
            sample['eye_reference'] = original_eye_texture_tensor
            sample['eye_bbox'] = eye_bbox_tensor  # [8] bbox

        # Add reference eye texture if requested (从参考图提取)
        if self.return_eye_reference:
            # Get current sample's prefix (subject ID)
            current_prefix = key
            current_index = index  # Frame index within the subject's sequence

            if current_prefix in self.prefix_to_indices:
                # Get all sample indices for this subject (excluding current sample)
                same_subject_indices = [i for i in self.prefix_to_indices[current_prefix] if i != idx]

                if len(same_subject_indices) > 0:
                    # Strategy 1: Search within temporal neighborhood (background consistency)
                    # Filter candidates within search radius for background consistency
                    candidate_indices = []

                    if self.reference_search_radius is not None:
                        # Search nearby frames (temporal proximity = same session/background)
                        for ref_idx in same_subject_indices:
                            ref_key, ref_index = self.index_to_query[ref_idx]

                            # Check if ref_key matches (same subject) and within temporal radius
                            if ref_key == current_prefix:
                                temporal_distance = abs(ref_index - current_index)
                                if temporal_distance <= self.reference_search_radius:
                                    candidate_indices.append(ref_idx)

                        # Fallback: If no candidates in neighborhood, expand to all frames
                        if len(candidate_indices) == 0:
                            candidate_indices = same_subject_indices
                    else:
                        # No radius constraint - search all frames of the subject
                        candidate_indices = same_subject_indices

                    # Strategy 2: Find the most forward-looking reference among candidates
                    best_ref_idx = None
                    min_angle_sum = float('inf')

                    for ref_idx in candidate_indices:
                        ref_key, ref_index = self.index_to_query[ref_idx]
                        ref_group = self.hdf[ref_key]

                        # Get gaze and head pose for this reference candidate
                        # Use ascontiguousarray to ensure contiguous memory from HDF5 slice
                        ref_norm_gaze = np.ascontiguousarray(ref_group['labels'][ref_index, 0:2], dtype=np.float32)  # [2]
                        ref_norm_head = np.ascontiguousarray(ref_group['labels'][ref_index, 2:4], dtype=np.float32)  # [2]

                        # Calculate total angle deviation (closer to 0 = more forward-looking)
                        gaze_angle = abs(ref_norm_gaze[0]) + abs(ref_norm_gaze[1])
                        head_angle = abs(ref_norm_head[0]) + abs(ref_norm_head[1])
                        angle_sum = gaze_angle + head_angle

                        if angle_sum < min_angle_sum:
                            min_angle_sum = angle_sum
                            best_ref_idx = ref_idx

                    # Load the best reference image
                    if best_ref_idx is not None:
                        ref_key, ref_index = self.index_to_query[best_ref_idx]
                        ref_group = self.hdf[ref_key]
                        ref_image = ref_group['pixels'][ref_index, ...]

                        # Generate reference mask from face_parsing (same logic as main mask)
                        if 'face_parsing' in ref_group:
                            ref_face_parsing = ref_group['face_parsing'][ref_index, ...]
                            ref_mask = np.zeros(ref_face_parsing.shape[:2], dtype=np.float32)

                            # Process left eye (label=1) and right eye (label=2)
                            for eye_label in [1, 2]:
                                eye_mask = (ref_face_parsing == eye_label).astype(np.uint8)
                                if eye_mask.sum() > 0:
                                    points = cv2.findNonZero(eye_mask)
                                    if points is not None and len(points) >= 5:
                                        (cx, cy), radius = cv2.minEnclosingCircle(points)
                                        cx, cy = int(cx), int(cy)
                                        radius = int(1.4 * radius)
                                        cv2.rectangle(ref_mask,
                                                    (cx - radius, cy - radius),
                                                    (cx + radius, cy + radius),
                                                    1, -1)
                        elif 'mask' in ref_group:
                            ref_mask = ref_group['mask'][ref_index, ...]


                        # Convert reference image to PIL
                        if isinstance(ref_image, np.ndarray):
                            if ref_image.dtype != np.uint8:
                                if ref_image.max() <= 1.0:
                                    ref_image = (ref_image * 255).astype(np.uint8)
                                else:
                                    ref_image = ref_image.astype(np.uint8)

                            if ref_image.ndim == 3 and ref_image.shape[2] == 3:
                                ref_image_pil = Image.fromarray(ref_image)
                            else:
                                raise ValueError(f"Unexpected reference image shape: {ref_image.shape}")

                        # Extract eye texture from reference
                        eye_texture_pil, ref_eye_bbox = self._extract_eye_texture_from_mask(
                            ref_image_pil, ref_mask, eye_size=256
                        )
                        # Transform to tensor (256×512 -> [3, 256, 512])
                        eye_texture_tensor = self.to_tensor(eye_texture_pil)
                        sample['eye_reference'] = eye_texture_tensor

                        # 添加参考图的眼睛 bbox
                        sample['eye_bbox'] = torch.tensor(ref_eye_bbox, dtype=torch.float32)

                        # Store reference gaze/head for debugging
                        # Use ascontiguousarray to ensure contiguous memory from HDF5 slice
                        ref_norm_gaze = np.ascontiguousarray(ref_group['labels'][ref_index, 0:2], dtype=np.float32)
                        ref_norm_head = np.ascontiguousarray(ref_group['labels'][ref_index, 2:4], dtype=np.float32)
                        sample['reference_gaze'] = torch.from_numpy(ref_norm_gaze)
                        sample['reference_head'] = torch.from_numpy(ref_norm_head)


        return sample

    def get_random_sample(self):
        """
        随机抽取一个样本，返回PIL图像、head_pose数据和掩码

        Returns:
            dict: 包含 'image_pil' (PIL.Image)、'head_pose' (list) 和 'mask' (numpy.ndarray) 的字典
        """
        # 随机选择一个索引
        random_idx = random.randint(0, len(self.index_to_query) - 1)

        # Lazy loading of HDF5 file
        if self.hdf is None:
            self.hdf = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True)

        # Get sample data
        key, index = self.index_to_query[random_idx]
        group = self.hdf[key]

        # Get original face image, labels and mask
        image = group['pixels'][index, ...]  # [H, W, 3], uint8 or float32
        norm_head = group['labels'][index, 2:4]  # [2], float32

        # Generate mask from face_parsing (same logic as __getitem__)
        if 'face_parsing' in group:
            face_parsing = group['face_parsing'][index, ...]
            mask = np.zeros(face_parsing.shape[:2], dtype=np.float32)

            # Process left eye (label=1) and right eye (label=2)
            for eye_label in [1, 2]:
                eye_mask = (face_parsing == eye_label).astype(np.uint8)
                if eye_mask.sum() > 0:
                    points = cv2.findNonZero(eye_mask)
                    if points is not None and len(points) >= 5:
                        (cx, cy), radius = cv2.minEnclosingCircle(points)
                        cx, cy = int(cx), int(cy)
                        radius = int(1.4 * radius)
                        cv2.rectangle(mask,
                                    (cx - radius, cy - radius),
                                    (cx + radius, cy + radius),
                                    1, -1)
        elif 'mask' in group:
            mask = group['mask'][index, ...]  # [H, W], mask data


        # Convert image to PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                # Convert to uint8 if it's float
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            # Convert to PIL Image (expects HWC format)
            if image.ndim == 3 and image.shape[2] == 3:
                image_pil = Image.fromarray(image)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")

        # Convert head pose to list
        head_pose = norm_head.tolist()

        # Convert mask to numpy array with float32 type
        if isinstance(mask, np.ndarray):
            mask = mask.astype(np.float32)
        else:
            # If it's not numpy array, convert it
            mask = np.array(mask, dtype=np.float32)

        print(f"📦 随机抽取样本: head_pose=[{head_pose[0]:.3f}, {head_pose[1]:.3f}], mask shape={mask.shape}")

        return {
            'image_pil': image_pil,
            'head_pose': head_pose,
            'mask': mask
        }

def draw_gaze(img, pitchyaw, thickness=2, color=(0, 255, 255)):
    """Draw gaze angle on given image with a given eye positions."""

    image_out = img
    (h, w) = img.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    im_g = np.array(image_out).copy()
    cv2.arrowedLine((im_g), tuple(np.round(pos).astype(np.int32)),
                tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                thickness, cv2.LINE_AA, tipLength=0.2)
    return im_g




if __name__ == "__main__":
    # Test code
    from torch.utils.data import DataLoader
    import torchvision.utils as vutils
    import os

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Test 1: Initialize dataset with DEFAULT JSON split path
    print("=" * 50)
    print("Test 1: Using DEFAULT JSON split file (No augmentation)")
    print("=" * 50)
    hdf_path = "/mnt/data/xhy/preprocess_sted/output_msted256/GazeCapture.h5"

    # Create train/val/test datasets - will use /home/xuhy/PycharmProjects/Unet-Gaze/gazecapture_split.json
    train_dataset = HDFDataset(hdf_path, split="train", aug=False)
    val_dataset = HDFDataset(hdf_path, split="val", aug=False)
    test_dataset = HDFDataset(hdf_path, split="test", aug=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Test 2: Test data augmentation
    print("\n" + "=" * 50)
    print("Test 2: Data Augmentation Enabled")
    print("=" * 50)
    train_dataset_aug = HDFDataset(hdf_path, split="train", aug=True)
    print(f"Train dataset (with aug) size: {len(train_dataset_aug)}")

    # Load a sample and verify augmentation is applied
    dataloader_aug = DataLoader(train_dataset_aug, batch_size=2, shuffle=True)
    batch_aug = next(iter(dataloader_aug))
    print("✅ Data augmentation test successful!")
    print(f"Augmented batch - Image shape: {batch_aug['image'].shape}")
    print(f"Augmented batch - Gaze: {batch_aug['norm_gaze']}")
    print(f"Augmented batch - Head: {batch_aug['norm_head']}")

    # Test 3: Use auto-detection
    print("\n" + "=" * 50)
    print("Test 3: Using auto-detected JSON file path")
    print("=" * 50)
    train_dataset_auto = HDFDataset(hdf_path, split="train", split_json_path="auto", aug=False)
    print(f"Train dataset size: {len(train_dataset_auto)}")

    # Test 4: Initialize dataset with random split (set split_json_path=None)
    print("\n" + "=" * 50)
    print("Test 4: Using random split (set split_json_path=None)")
    print("=" * 50)
    train_dataset_random = HDFDataset(hdf_path, split="train", split_json_path=None, split_ratio=(0.8, 0.1, 0.1), seed=42, aug=False)
    val_dataset_random = HDFDataset(hdf_path, split="val", split_json_path=None, split_ratio=(0.8, 0.1, 0.1), seed=42, aug=False)
    test_dataset_random = HDFDataset(hdf_path, split="test", split_json_path=None, split_ratio=(0.8, 0.1, 0.1), seed=42, aug=False)

    print(f"Train dataset size: {len(train_dataset_random)}")
    print(f"Val dataset size: {len(val_dataset_random)}")
    print(f"Test dataset size: {len(test_dataset_random)}")

    # Test 5: Loading a batch
    print("\n" + "=" * 50)
    print("Test 5: Loading a batch (no augmentation)")
    print("=" * 50)
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    for batch in dataloader:
        # Print information
        print("Batch keys:", batch.keys())
        print("Image shape:", batch['image'].shape)
        print("Mask shape:", batch['mask'].shape)
        print("Norm gaze:", batch['norm_gaze'])
        print("Norm head:", batch['norm_head'])
        break

