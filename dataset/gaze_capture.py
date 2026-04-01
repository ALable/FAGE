"""
FAGE — Paired Frame Dataset for Eye-Only Gaze Generation
从 Unet-Gaze/dataset/gaze_capture.py 精简而来。

返回配对帧: source 提供眼部外观, target 提供 gaze 方向和 GT。
"""
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

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class HDFDataset(Dataset):
    """GazeCapture HDF5 数据集，支持配对帧 + 眼部 crop

    每个 __getitem__ 返回同一用户的相邻帧配对 (source + target)。
    """

    def __init__(self, hdf_file_path,
                 prefixes=None,
                 split="train",
                 split_ratio=(0.8, 0.1, 0.1),
                 seed=42,
                 split_json_path=None,
                 frame_offset_range=2,
                 eye_crop_size=(64, 64),
                 eye_expand_ratio=1.5,
                 transform=None,
                 pick_at_least_per_person=2):
        """
        Args:
            hdf_file_path: HDF5 文件路径
            split: 'train', 'val', 'test'
            split_ratio: 训练/验证/测试比例
            seed: 随机种子
            split_json_path: 预定义 split JSON 路径 (None=随机 split)
            frame_offset_range: 相邻帧偏移范围 ±N
            eye_crop_size: 每只眼 crop 尺寸 (H, W)
            eye_expand_ratio: 眼部 bbox 扩展倍率
            transform: 自定义 transform (默认 ToTensor + Normalize)
            pick_at_least_per_person: 每个用户最少帧数 (配对需要至少2帧)
        """
        assert os.path.isfile(hdf_file_path), f"HDF5 not found: {hdf_file_path}"
        self.hdf_path = hdf_file_path
        self.hdf = None
        self.split = split
        self.frame_offset_range = frame_offset_range
        self.eye_crop_size = eye_crop_size
        self.eye_expand_ratio = eye_expand_ratio

        if transform is None:
            self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.to_tensor = transform

        # 加载数据集信息
        with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as h5f:
            hdf_keys = sorted(list(h5f.keys()))
            if prefixes is None:
                self.prefixes = hdf_keys
            else:
                self.prefixes = [k for k in prefixes if k in h5f]

            # 过滤帧数不足的用户 (配对需要至少2帧)
            min_frames = max(pick_at_least_per_person or 2, 2)
            self.prefixes = [k for k in self.prefixes
                             if k in h5f and len(next(iter(h5f[k].values()))) >= min_frames]

            # Split
            if split_json_path and os.path.isfile(split_json_path):
                with open(split_json_path, 'r') as f:
                    all_splits = json.load(f)
                if self.split not in all_splits:
                    raise ValueError(f"Split '{self.split}' not in JSON. Available: {list(all_splits.keys())}")
                json_prefixes = all_splits[self.split]
                self.prefixes = [k for k in json_prefixes if k in self.prefixes]
                logging.info(f"JSON split '{self.split}': {len(self.prefixes)} subjects")
            else:
                rng = np.random.RandomState(seed)
                num = len(self.prefixes)
                idxs = np.arange(num)
                rng.shuffle(idxs)
                n_train = int(split_ratio[0] * num)
                n_val = int(split_ratio[1] * num)
                if self.split == "train":
                    self.prefixes = [self.prefixes[i] for i in idxs[:n_train]]
                elif self.split == "val":
                    self.prefixes = [self.prefixes[i] for i in idxs[n_train:n_train+n_val]]
                elif self.split == "test":
                    self.prefixes = [self.prefixes[i] for i in idxs[n_train+n_val:]]

            # 构建 index → (prefix, frame_idx) 映射
            self.index_to_query = sum([
                [(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                for prefix in self.prefixes
            ], [])

            # 构建 prefix → [全局 indices] 映射 (用于配对帧选择)
            self.prefix_to_indices = {}
            for idx, (prefix, _) in enumerate(self.index_to_query):
                if prefix not in self.prefix_to_indices:
                    self.prefix_to_indices[prefix] = []
                self.prefix_to_indices[prefix].append(idx)

            # 构建 prefix → subject_idx 映射
            self.prefix_to_subject_idx = {p: i for i, p in enumerate(sorted(self.prefix_to_indices.keys()))}

        logging.info(f"FAGE HDFDataset [{self.split}]: {len(self.index_to_query)} frames, "
                     f"{len(self.prefix_to_indices)} subjects")

    def __len__(self):
        return len(self.index_to_query)

    def _crop_eyes(self, image_np, face_parsing):
        """裁剪左右眼区域并拼接为 [3, crop_h, crop_w*2]

        Args:
            image_np: [H, W, 3] uint8
            face_parsing: [H, W] uint8 (1=left_eye, 2=right_eye)

        Returns:
            eye_crops: [3, 64, 128] tensor (左眼64x64 | 右眼64x64)
            eye_bbox: [8] float tensor 归一化坐标 [lx1,ly1,lx2,ly2, rx1,ry1,rx2,ry2]
        """
        H, W = image_np.shape[:2]
        crop_h, crop_w = self.eye_crop_size
        eye_crops = []
        bbox_list = []

        for eye_label in [1, 2]:  # 1=left, 2=right
            eye_mask = (face_parsing == eye_label).astype(np.uint8)
            if eye_mask.sum() > 10:
                points = cv2.findNonZero(eye_mask)
                if points is not None and len(points) >= 5:
                    (cx, cy), radius = cv2.minEnclosingCircle(points)
                    r = int(radius * self.eye_expand_ratio)
                    x1 = max(0, int(cx) - r)
                    y1 = max(0, int(cy) - r)
                    x2 = min(W, int(cx) + r)
                    y2 = min(H, int(cy) + r)

                    crop = image_np[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_resized = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
                        eye_crops.append(crop_resized)
                        bbox_list.extend([x1/W, y1/H, x2/W, y2/H])
                        continue

            # fallback: 默认 bbox 和黑色 crop
            eye_crops.append(np.zeros((crop_h, crop_w, 3), dtype=np.uint8))
            bbox_list.extend([0, 0, 0, 0])

        # 拼接左右眼: [crop_h, crop_w*2, 3]
        combined = np.concatenate(eye_crops, axis=1)
        combined_tensor = self.to_tensor(Image.fromarray(combined))  # [3, crop_h, crop_w*2]
        bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)
        return combined_tensor, bbox_tensor

    def _load_frame(self, key, index):
        """加载单帧数据

        Returns:
            image_np: [H, W, 3] uint8
            face_parsing: [H, W] uint8
            gaze: [2] float32
            head: [2] float32
        """
        group = self.hdf[key]
        image = group['pixels'][index, ...]
        gaze = np.ascontiguousarray(group['labels'][index, 0:2], dtype=np.float32)
        head = np.ascontiguousarray(group['labels'][index, 2:4], dtype=np.float32)

        face_parsing = None
        if 'face_parsing' in group:
            face_parsing = group['face_parsing'][index, ...]

        # 确保 uint8
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

        return image, face_parsing, gaze, head

    def __getitem__(self, idx):
        """返回配对帧 (source + target)

        Returns:
            dict with:
                source_image: [3, 256, 256]
                source_gaze, source_head: [2]
                source_eye_crops: [3, 64, 128]
                source_eye_bbox: [8]
                target_image: [3, 256, 256]
                target_gaze, target_head: [2]
                target_eye_crops: [3, 64, 128]
                target_eye_bbox: [8]
                subject_idx: int
        """
        if self.hdf is None:
            self.hdf = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True)

        # === Source 帧 ===
        key, index = self.index_to_query[idx]
        source_np, source_parsing, source_gaze, source_head = self._load_frame(key, index)

        # === Target 帧 (同一用户的相邻帧) ===
        group = self.hdf[key]
        num_frames = group['pixels'].shape[0]

        # 在 ±offset_range 内随机选取 (避免选到自身)
        offsets = list(range(-self.frame_offset_range, self.frame_offset_range + 1))
        offsets = [o for o in offsets if o != 0]
        offset = random.choice(offsets)
        target_index = max(0, min(num_frames - 1, index + offset))
        if target_index == index:
            target_index = (index + 1) % num_frames

        target_np, target_parsing, target_gaze, target_head = self._load_frame(key, target_index)

        # === Eye crops ===
        source_eye_crops, source_eye_bbox = self._crop_eyes(source_np, source_parsing)
        target_eye_crops, target_eye_bbox = self._crop_eyes(target_np, target_parsing)

        # === Image tensors ===
        source_image = self.to_tensor(Image.fromarray(source_np))
        target_image = self.to_tensor(Image.fromarray(target_np))

        # === Subject index ===
        subject_idx = self.prefix_to_subject_idx.get(key, 0)

        return {
            'source_image': source_image,
            'source_gaze': torch.from_numpy(source_gaze),
            'source_head': torch.from_numpy(source_head),
            'source_eye_crops': source_eye_crops,
            'source_eye_bbox': source_eye_bbox,

            'target_image': target_image,
            'target_gaze': torch.from_numpy(target_gaze),
            'target_head': torch.from_numpy(target_head),
            'target_eye_crops': target_eye_crops,
            'target_eye_bbox': target_eye_bbox,

            'subject_idx': subject_idx,
        }

    @property
    def num_subjects(self):
        return len(self.prefix_to_subject_idx)
