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
                 input_eye_crop_size=None,
                 input_eye_expand_ratio=None,
                 transform=None,
                 pick_at_least_per_person=2,
                 max_head_diff=None,
                 min_gaze_diff=None,
                 max_target_retries=10):
        """
        Args:
            hdf_file_path: HDF5 文件路径
            split: 'train', 'val', 'test'
            split_ratio: 训练/验证/测试比例
            seed: 随机种子
            split_json_path: 预定义 split JSON 路径 (None=随机 split)
            frame_offset_range: 相邻帧偏移范围 ±N（无过滤时生效）
            eye_crop_size: 每只眼 crop 尺寸 (H, W)
            eye_expand_ratio: 眼部 bbox 扩展倍率
            transform: 自定义 transform (默认 ToTensor + Normalize)
            pick_at_least_per_person: 每个用户最少帧数 (配对需要至少2帧)
            max_head_diff: 头姿差异上限 (rad)，超过则重新选取 target；None=不过滤
            min_gaze_diff: 注视差异下限 (rad)，在满足头姿条件的候选中优先选差异大的；None=随机选
            max_target_retries: 过滤时最多尝试的候选帧数（从全片段随机采样）
        """
        assert os.path.isfile(hdf_file_path), f"HDF5 not found: {hdf_file_path}"
        self.hdf_path = hdf_file_path
        self.hdf = None
        self.split = split
        self.frame_offset_range = frame_offset_range
        self.max_head_diff       = max_head_diff
        self.min_gaze_diff       = min_gaze_diff
        self.max_target_retries  = max_target_retries
        self.eye_crop_size = eye_crop_size
        self.eye_expand_ratio = eye_expand_ratio
        # 模型输入尺寸（比监督尺寸更大，默认与监督相同退化）
        self.input_eye_crop_size    = input_eye_crop_size   if input_eye_crop_size   is not None else eye_crop_size
        self.input_eye_expand_ratio = input_eye_expand_ratio if input_eye_expand_ratio is not None else eye_expand_ratio

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
        """裁剪左右眼，同时返回 input（大范围）和 target（紧范围）两套 crop

        Args:
            image_np: [H, W, 3] uint8
            face_parsing: [H, W] uint8 (1=left_eye, 2=right_eye)

        Returns:
            input_tensor:  [3, in_h, in_w*2]   大范围，模型输入（width concat）
            target_tensor: [3, crop_h, crop_w*2] 紧范围，loss 监督（width concat）
            eye_bbox:      [8] float tensor 归一化坐标（基于 tight bbox，用于 paste_eyes）
        """
        H, W = image_np.shape[:2]
        crop_h, crop_w = self.eye_crop_size
        in_h,   in_w   = self.input_eye_crop_size

        input_crops  = []
        target_crops = []
        bbox_list    = []

        for eye_label in [1, 2]:  # 1=left, 2=right
            eye_mask = (face_parsing == eye_label).astype(np.uint8)
            if eye_mask.sum() > 10:
                points = cv2.findNonZero(eye_mask)
                if points is not None and len(points) >= 5:
                    (cx, cy), radius = cv2.minEnclosingCircle(points)

                    # tight crop（监督用，eye_expand_ratio）
                    r_t = int(radius * self.eye_expand_ratio)
                    tx1 = max(0, int(cx) - r_t)
                    ty1 = max(0, int(cy) - r_t)
                    tx2 = min(W, int(cx) + r_t)
                    ty2 = min(H, int(cy) + r_t)
                    tight_crop = image_np[ty1:ty2, tx1:tx2]

                    # large crop（输入用，input_eye_expand_ratio）
                    r_i = int(radius * self.input_eye_expand_ratio)
                    ix1 = max(0, int(cx) - r_i)
                    iy1 = max(0, int(cy) - r_i)
                    ix2 = min(W, int(cx) + r_i)
                    iy2 = min(H, int(cy) + r_i)
                    large_crop = image_np[iy1:iy2, ix1:ix2]

                    if tight_crop.size > 0 and large_crop.size > 0:
                        target_crops.append(cv2.resize(tight_crop, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4))
                        input_crops.append( cv2.resize(large_crop, (in_w,   in_h),   interpolation=cv2.INTER_LANCZOS4))
                        bbox_list.extend([tx1/W, ty1/H, tx2/W, ty2/H])
                        continue

            # fallback
            target_crops.append(np.zeros((crop_h, crop_w, 3), dtype=np.uint8))
            input_crops.append( np.zeros((in_h,   in_w,   3), dtype=np.uint8))
            bbox_list.extend([0, 0, 0, 0])

        # 宽度拼接左右眼 (width concat)
        input_tensor  = torch.cat([self.to_tensor(Image.fromarray(input_crops[0])),
                                   self.to_tensor(Image.fromarray(input_crops[1]))],  dim=2)  # [3, in_h, in_w*2]
        target_tensor = torch.cat([self.to_tensor(Image.fromarray(target_crops[0])),
                                   self.to_tensor(Image.fromarray(target_crops[1]))], dim=2)  # [3, crop_h, crop_w*2]
        bbox_tensor   = torch.tensor(bbox_list, dtype=torch.float32)
        return input_tensor, target_tensor, bbox_tensor

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

    def _pick_target_index(self, key, source_index, source_gaze, source_head, num_frames):
        """从 ±frame_offset_range 内的候选帧中，选出头姿接近且注视差异最大的 target

        策略:
          1. 枚举 ±frame_offset_range 内所有合法候选（不含自身，去重）
          2. 一次 fancy-index 读取所有候选的 labels（只读标签，不读像素）
          3. 若开启过滤：筛选 head_diff ≤ max_head_diff 的合格帧；无合格帧时回退到 head_diff 最小者
          4. 在合格帧中选 gaze_diff 最大的
        """
        # 枚举候选 indices，去重、去掉退化到 source 自身的
        seen = set()
        candidates = []
        for o in range(-self.frame_offset_range, self.frame_offset_range + 1):
            if o == 0:
                continue
            tidx = max(0, min(num_frames - 1, source_index + o))
            if tidx != source_index and tidx not in seen:
                seen.add(tidx)
                candidates.append(tidx)

        if not candidates:
            return (source_index + 1) % num_frames

        if self.max_head_diff is None:
            # 无过滤：随机选一个（兼容原行为）
            return random.choice(candidates)

        # 一次 fancy-index 读全部候选的 labels，避免逐帧 IO
        cand_arr = np.array(candidates)
        cand_labels = self.hdf[key]['labels'][cand_arr]          # [K, 4]
        cand_gaze   = cand_labels[:, 0:2].astype(np.float32)    # [K, 2]
        cand_head   = cand_labels[:, 2:4].astype(np.float32)    # [K, 2]

        head_diffs = np.linalg.norm(cand_head - source_head, axis=1)  # [K]
        gaze_diffs = np.linalg.norm(cand_gaze - source_gaze, axis=1)  # [K]

        valid_mask = head_diffs <= self.max_head_diff
        if not valid_mask.any():
            # 无合格候选：回退到 head_diff 最小的
            return candidates[int(np.argmin(head_diffs))]

        # 合格帧中选 gaze_diff 最大的
        masked_gaze = np.where(valid_mask, gaze_diffs, -1.0)
        return candidates[int(np.argmax(masked_gaze))]

    def __getitem__(self, idx):
        """返回配对帧 (source + target)

        Returns:
            dict with:
                source_image:             [3, H, W]        全脸图像
                source_gaze, source_head: [2]              注视/头姿角度 (rad)
                source_input_eye_crops:   [3, in_h, in_w*2]  大范围眼部 crop，模型输入（width concat）
                source_eye_crops:         [3, h, w*2]        紧范围眼部 crop，GT 备用（width concat）
                source_eye_bbox:          [8]              归一化 bbox 坐标 (tight)
                target_image:             [3, H, W]        全脸图像
                target_gaze, target_head: [2]              注视/头姿角度 (rad)
                target_input_eye_crops:   [3, in_h, in_w*2]  大范围眼部 crop（暂未用）
                target_eye_crops:         [3, h, w*2]        紧范围眼部 crop，loss 监督（width concat）
                target_eye_bbox:          [8]              归一化 bbox 坐标 (tight)
                subject_idx:              int              用户编号
        """
        if self.hdf is None:
            self.hdf = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True)

        # === Source 帧 ===
        key, index = self.index_to_query[idx]
        source_np, source_parsing, source_gaze, source_head = self._load_frame(key, index)

        # === Target 帧：头姿接近、注视差异大 ===
        group = self.hdf[key]
        num_frames = group['pixels'].shape[0]
        target_index = self._pick_target_index(key, index, source_gaze, source_head, num_frames)

        target_np, target_parsing, target_gaze, target_head = self._load_frame(key, target_index)

        # === Eye crops ===
        src_input, src_target, source_eye_bbox = self._crop_eyes(source_np, source_parsing)
        tgt_input, tgt_target, target_eye_bbox = self._crop_eyes(target_np, target_parsing)

        # === Image tensors ===
        source_image = self.to_tensor(Image.fromarray(source_np))
        target_image = self.to_tensor(Image.fromarray(target_np))

        # === Subject index ===
        subject_idx = self.prefix_to_subject_idx.get(key, 0)

        return {
            'source_image': source_image,
            'source_gaze': torch.from_numpy(source_gaze),
            'source_head': torch.from_numpy(source_head),
            'source_input_eye_crops': src_input,   # [3, in_h, in_w*2]  大范围，模型输入（width concat）
            'source_eye_crops':       src_target,  # [3, h, w*2]        紧范围，GT（备用）
            'source_eye_bbox': source_eye_bbox,

            'target_image': target_image,
            'target_gaze': torch.from_numpy(target_gaze),
            'target_head': torch.from_numpy(target_head),
            'target_input_eye_crops': tgt_input,   # [3, in_h, in_w*2]  大范围（暂未用）
            'target_eye_crops':       tgt_target,  # [3, h, w*2]        紧范围，loss 监督（width concat）
            'target_eye_bbox': target_eye_bbox,

            'subject_idx': subject_idx,
        }

    @property
    def num_subjects(self):
        return len(self.prefix_to_subject_idx)
