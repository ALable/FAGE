import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import random
import logging
from PIL import Image
from torchvision import transforms
from io import BytesIO
import lmdb
import re
import matplotlib.pyplot as plt
import threading
import pickle
import hashlib
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Thread-local storage for LMDB environments
_thread_local = threading.local()
def get_thread_local_env(lmdb_path):
    """Get thread-local LMDB environment for multi-process safety"""
    if not hasattr(_thread_local, 'env') or _thread_local.env is None:
        _thread_local.env = lmdb.open(
            lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
    return _thread_local.env

class ETHXGazeLMDBDataset(Dataset):
    """
    ETH-XGAZE LMDB数据集加载器
    专注于LMDB数据读取和眼部patch处理
    """

    # 预编译正则表达式(类级别,避免重复编译)
    _GAZE_PATTERN = re.compile(
        r'-([+-]?\d+\.?\d*)_([+-]?\d+\.?\d*)-([+-]?\d+\.?\d*)_([+-]?\d+\.?\d*)-s'
    )
    _SUBJECT_PATTERN = re.compile(r'-s(\d+)')

    def __init__(self, lmdb_path,
                 transform=None,
                 resolution=256,
                 aug=False,
                 split="train",
                 split_ratio=(0.8, 0.1, 0.1),
                 seed=41,
                 eye_scale_factor=1.2,
                 use_precomputed_masks=True,
                 convert_to_square=False,
                 return_reference=True,
                 identity_model='ArcFace'):
        """
        初始化ETH-XGAZE LMDB数据集

        Args:
            lmdb_path: LMDB数据库路径
            transform: 图像变换
            resolution: 图像分辨率
            aug: 是否使用数据增强
            split: 'train', 'val', or 'test'
            split_ratio: 数据分割比例
            seed: 随机种子
            eye_scale_factor: 眼部区域扩大倍数，默认1.2
            use_precomputed_masks: 是否使用预计算的眼部掩码，默认True
            convert_to_square: 是否将预计算的椭圆掩码转换为正方形掩码，默认False
            return_reference: 是否返回同一人物的reference图像及identity特征，默认True
            identity_model:
        """
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.resolution = resolution
        self.aug = aug
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.eye_scale_factor = eye_scale_factor
        self.use_precomputed_masks = use_precomputed_masks
        self.convert_to_square = convert_to_square
        self.return_reference = return_reference
        # self.identity_model = identity_model

        # 缓存：预构建索引
        self._image_keys_cache = {}
        self._gaze_info_cache = {}
        self._eyemask_keys_cache = {}
        self._subject_to_indices = {}  # 人物ID到数据索引的映射

        # 获取数据长度并检查预计算掩码，构建索引
        self._initialize_dataset()

        # 数据分割
        self._split_data()
        # 默认图像变换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            ])

    def _get_cache_path(self):
        """生成索引缓存文件路径

        基于 lmdb_path 和关键配置参数生成唯一的缓存文件名,
        确保配置改变时缓存失效
        """
        # 生成缓存键(包含影响索引的所有参数)
        cache_key = f"{self.lmdb_path}_{self.use_precomputed_masks}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]

        # 缓存文件放在 LMDB 数据库同目录
        cache_file = os.path.join(
            os.path.dirname(self.lmdb_path),
            f".index_cache_{cache_hash}.pkl"
        )
        return cache_file

    def _initialize_dataset(self):
        """初始化数据集，构建索引缓存(支持磁盘缓存加速)"""
        cache_file = self._get_cache_path()

        # 尝试加载缓存
        if os.path.exists(cache_file):
            logging.info(f"🚀 发现索引缓存: {cache_file}")
            try:
                import time
                start_time = time.time()
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self._image_keys_cache = cache_data['image_keys']
                    self._gaze_info_cache = cache_data['gaze_info']
                    self._eyemask_keys_cache = cache_data['eyemask_keys']
                    self._subject_to_indices = cache_data['subject_to_indices']
                    self.total_length = cache_data['total_length']
                    self.has_precomputed_masks = cache_data['has_precomputed_masks']
                load_time = time.time() - start_time
                logging.info(f"✅ 索引缓存加载完成: {len(self._image_keys_cache)} 样本 "
                           f"(耗时 {load_time:.2f}秒)")
                logging.info(f"📊 发现 {len(self._subject_to_indices)} 个不同的人物ID")
                return
            except Exception as e:
                logging.warning(f"⚠️  缓存加载失败,重新构建索引: {e}")

        # 缓存不存在或加载失败,执行原始的索引构建
        logging.info("🔍 正在构建数据集索引(首次运行较慢,后续会使用缓存)...")
        import time
        start_time = time.time()

        env = lmdb.open(
            self.lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with env.begin(write=False) as txn:
            # 检查是否有预计算的眼部掩码
            eyemask_length_data = txn.get('eyemask-length'.encode('utf-8'))
            if eyemask_length_data and self.use_precomputed_masks:
                self.has_precomputed_masks = True
                logging.info("✅ 检测到预计算眼部掩码，将使用高速模式")
            else:
                self.has_precomputed_masks = False
                if self.use_precomputed_masks:
                    logging.warning("⚠️ 未找到预计算眼部掩码，将使用实时计算模式")
                    logging.warning(
                        "   建议运行 preprocess_eth_xgaze_masks.py 预处理数据")

            # 获取数据总长度
            length_data = txn.get('label-length'.encode('utf-8'))
            if length_data is None:
                # 通过扫描键名确定大小
                cursor = txn.cursor()
                max_index = -1
                for key, value in cursor:
                    key_str = key.decode('utf-8')
                    if key_str.startswith('image-'):
                        parts = key_str.split('-')
                        if len(parts) >= 2 and parts[1].isdigit():
                            index = int(parts[1])
                            max_index = max(max_index, index)
                self.total_length = max_index + 1
            else:
                self.total_length = int(length_data.decode('utf-8'))

            # 构建索引缓存
            logging.info(f"📊 构建 {self.total_length} 个样本的索引缓存...")
            cursor = txn.cursor()

            for key, value in cursor:
                key_str = key.decode('utf-8')

                # 处理 image 键
                if key_str.startswith('image-'):
                    data_idx = self._extract_data_idx_from_key(key_str)
                    if data_idx is not None:
                        self._image_keys_cache[data_idx] = key_str
                        # 同时解析并缓存 gaze 信息
                        gaze_info = self._parse_gaze_info_from_key(key_str)
                        if gaze_info:
                            self._gaze_info_cache[data_idx] = gaze_info

                        # 提取并缓存人物ID映射
                        subject_id = self._extract_subject_id_from_key(key_str)
                        if subject_id:
                            if subject_id not in self._subject_to_indices:
                                self._subject_to_indices[subject_id] = []
                            self._subject_to_indices[subject_id].append(
                                data_idx)

                # 处理 eyemask 键
                elif key_str.startswith('eyemask-') and not key_str.endswith(('-length', '-metadata')):
                    data_idx = self._extract_data_idx_from_key(key_str)
                    if data_idx is not None:
                        self._eyemask_keys_cache[data_idx] = key_str

        env.close()

        build_time = time.time() - start_time
        logging.info(
            f"✅ 索引缓存构建完成: {len(self._image_keys_cache)} 个图像, "
            f"{len(self._eyemask_keys_cache)} 个掩码 (耗时 {build_time:.2f}秒)")
        logging.info(f"📊 发现 {len(self._subject_to_indices)} 个不同的人物ID")

        # 保存缓存到磁盘
        try:
            logging.info(f"💾 保存索引缓存: {cache_file}")
            cache_data = {
                'image_keys': self._image_keys_cache,
                'gaze_info': self._gaze_info_cache,
                'eyemask_keys': self._eyemask_keys_cache,
                'subject_to_indices': self._subject_to_indices,
                'total_length': self.total_length,
                'has_precomputed_masks': self.has_precomputed_masks
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("索引缓存已保存")
        except Exception as e:
            logging.warning(f"缓存保存失败: {e}")

    def _extract_data_idx_from_key(self, key_str):
        """从键名中提取数据索引"""
        parts = key_str.split('-')
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        return None

    def _extract_subject_id_from_key(self, key_str):
        """Extract subject ID from key string.

        Args:
            key_str: Key string like 'image-0000003--0.4652_0.2783-0.2717_0.0938-s0000'

        Returns:
            str: Subject ID like 's0000' or None if not found
        """
        try:
            # Extract subject ID using pre-compiled regex
            match = self._SUBJECT_PATTERN.search(key_str)
            if match:
                return f"s{match.group(1)}"
            else:
                logging.warning(
                    f"Failed to extract subject ID from key '{key_str}'")
                return None
        except Exception as e:
            logging.warning(
                f"Error extracting subject ID from key '{key_str}': {e}")
            return None

    def _parse_gaze_info_from_key(self, key_str):
        """Parses gaze information from the key string.

        Key format: {prefix}-{index}-{gaze_x}_{gaze_y}-{head_x}_{head_y}-s{subject}
        Example: eyemask-0000003--0.4652_0.2783--0.2717_0.0938-s0000
        """
        try:
            gaze_x, gaze_y, head_x, head_y = 0.0, 0.0, 0.0, 0.0
            # 键名格式: eyemask-{idx}-{gaze_pitch}_{gaze_yaw}-{head_pitch}_{head_yaw}-s{subject}
            # 例如: eyemask-0000829--0.2561_0.2679--0.1681_0.0314-s0004
            # 使用预编译的正则表达式
            match = self._GAZE_PATTERN.search(key_str)
            if match:
                gaze_x = float(match.group(1))
                gaze_y = float(match.group(2))
                head_x = float(match.group(3))
                head_y = float(match.group(4))
            else:
                logging.error(
                    f"Failed to extract gaze/head info using regex for key '{key_str}'. Defaulting to 0.0.")

            return {
                'gaze_x': gaze_x,
                'gaze_y': gaze_y,
                'head_x': head_x,
                'head_y': head_y
            }

        except (ValueError, IndexError) as e:
            logging.warning(f"Failed to parse key '{key_str}': {e}")
            return {'gaze_x': 0.0, 'gaze_y': 0.0, 'head_x': 0.0, 'head_y': 0.0}

    def _split_data(self):
        """Splits data into train, validation, and test sets by subject ID."""
        # 获取所有人物ID并按人物进行划分
        subject_ids = list(self._subject_to_indices.keys())
        subject_ids.sort()  # 确保一致性

        rng = np.random.RandomState(self.seed)
        rng.shuffle(subject_ids)  # 随机打乱人物ID顺序

        n_subjects = len(subject_ids)
        n_train_subjects = int(self.split_ratio[0] * n_subjects)
        n_val_subjects = int(self.split_ratio[1] * n_subjects)

        # 按人物ID划分
        if self.split == "train":
            selected_subjects = subject_ids[:n_train_subjects]
        elif self.split == "val":
            # selected_subjects = subject_ids[n_train_subjects:n_train_subjects+n_val_subjects]
            selected_subjects=subject_ids[n_train_subjects:]
        else:  # test
            selected_subjects = subject_ids[n_train_subjects+n_val_subjects:]

        # 收集选定人物的所有数据索引
        self.indices = []
        for subject_id in selected_subjects:
            self.indices.extend(self._subject_to_indices[subject_id])

        self.indices = np.array(self.indices)

        # 日志信息
        logging.info(f"📊 数据集划分 ({self.split}):")
        logging.info(f"   - 选择人物数: {len(selected_subjects)}/{n_subjects}")
        logging.info(f"   - 数据样本数: {len(self.indices)}")
        if len(selected_subjects) <= 10:  # 只在人物数较少时显示具体ID
            logging.info(f"   - 人物ID: {selected_subjects}")

    def _get_precomputed_eye_mask(self, txn, data_idx):
        """获取预计算的精确眼睛掩码 根据人脸解析图得到"""
        if not self.has_precomputed_masks:
            return None

        # 从缓存中直接获取键名
        eyemask_key = self._eyemask_keys_cache.get(data_idx)
        if eyemask_key:
            return txn.get(eyemask_key.encode('utf-8'))
        return None

    def _expand_ellipse_mask(self, ellipse_mask):
        """
        扩张椭圆形掩码（应用 eye_scale_factor）
        
        Args:
            ellipse_mask: 椭圆形掩码 (numpy array)
            
        Returns:
            expanded_mask: 扩张后的椭圆形掩码 (numpy array)
        """
        if isinstance(ellipse_mask, Image.Image):
            mask_array = np.array(ellipse_mask, dtype=np.uint8)
        else:
            mask_array = ellipse_mask.astype(np.uint8)
        
        h, w = mask_array.shape[:2]
        expanded_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 二值化掩码
        _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
        
        # 找到所有连通区域
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return mask_array
        
        # 对每个椭圆区域进行扩张
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # 过滤小区域
                # 拟合椭圆
                if len(contour) >= 5:  # fitEllipse 需要至少5个点
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    
                    # 应用缩放因子
                    expanded_axes = (
                        axes[0] * self.eye_scale_factor,
                        axes[1] * self.eye_scale_factor
                    )
                    
                    # 绘制扩张后的椭圆
                    cv2.ellipse(expanded_mask, 
                               (int(center[0]), int(center[1])),
                               (int(expanded_axes[0]/2), int(expanded_axes[1]/2)),
                               angle, 0, 360, 255, -1)
        
        return expanded_mask
    
    def _convert_ellipse_mask_to_square(self, ellipse_mask):
        """
        将椭圆形掩码转换为正方形掩码

        Args:
            ellipse_mask: 椭圆形掩码 (numpy array)

        Returns:
            square_mask: 正方形掩码 (numpy array)
        """
        if isinstance(ellipse_mask, Image.Image):
            mask_array = np.array(ellipse_mask, dtype=np.uint8)
        else:
            mask_array = ellipse_mask.astype(np.uint8)

        h, w = mask_array.shape[:2]
        square_mask = np.zeros((h, w), dtype=np.uint8)

        # 二值化掩码
        _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)

        # 找到所有连通区域
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            # 如果没有找到轮廓，返回原掩码
            return mask_array

        # 收集眼部区域信息
        eye_info = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # 过滤小区域
                # 计算最小外接圆
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                eye_info.append({
                    'center': (int(center_x), int(center_y)),
                    'radius': radius,
                    'area': area
                })

        if len(eye_info) == 0:
            return mask_array

        # 按面积排序，取最大的两个（左右眼）
        eye_info = sorted(eye_info, key=lambda x: x['area'], reverse=True)[:2]
        # 按x坐标排序（从左到右）
        eye_info = sorted(eye_info, key=lambda x: x['center'][0])

        # 为每只眼睛创建正方形区域（应用eye_scale_factor）
        for eye in eye_info:
            center_x, center_y = eye['center']
            radius = eye['radius']

            # 应用缩放因子并计算正方形边长
            expanded_radius = radius * self.eye_scale_factor
            side_length = int(expanded_radius * 2)
            half_side = side_length // 2

            # 计算正方形边界
            x1 = max(0, center_x - half_side)
            y1 = max(0, center_y - half_side)
            x2 = min(w, x1 + side_length)
            y2 = min(h, y1 + side_length)

            # 填充正方形区域
            square_mask[y1:y2, x1:x2] = 255

        return square_mask

    def _apply_augmentation(self, image, mask, gaze_info):
        """
        应用数据增强到图像、掩码和视线信息

        Args:
            image: PIL Image - 输入图像
            mask: numpy array - 眼部掩码
            gaze_info: dict - 视线和头部姿态信息

        Returns:
            image_aug: PIL Image - 增强后的图像
            mask_aug: numpy array - 增强后的掩码
            gaze_info_aug: dict - 调整后的视线信息
        """
        gaze_info_aug = gaze_info.copy()
        # Ensure mask is a contiguous copy to avoid storage resize issues
        mask = np.ascontiguousarray(mask)

        # 1. 随机水平翻转 (50%概率)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.fliplr(mask).copy()  # .copy() 避免负stride问题
            # 水平翻转需要反转yaw角度（左右方向）
            gaze_info_aug['gaze_y'] = -gaze_info['gaze_y']
            gaze_info_aug['head_y'] = -gaze_info['head_y']

        # 2. 随机颜色抖动
        if random.random() < 0.8:
            # 亮度抖动 ±20%
            brightness_factor = random.uniform(0.8, 1.2)
            image = transforms.functional.adjust_brightness(image, brightness_factor)

            # 对比度抖动 ±20%
            contrast_factor = random.uniform(0.8, 1.2)
            image = transforms.functional.adjust_contrast(image, contrast_factor)

            # 饱和度抖动 ±20%
            saturation_factor = random.uniform(0.8, 1.2)
            image = transforms.functional.adjust_saturation(image, saturation_factor)

            # 色调抖动 ±10度
            hue_factor = random.uniform(-0.05, 0.05)
            image = transforms.functional.adjust_hue(image, hue_factor)

        # 3. 随机旋转 (±5度)
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
            # 旋转掩码（使用最近邻插值保持二值性）
            mask_pil = Image.fromarray(mask)
            mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST, fillcolor=0)
            # 确保旋转后的 mask 是连续的内存
            mask = np.ascontiguousarray(np.array(mask_pil))

            # 旋转角度会轻微影响gaze，但在小角度范围内可以忽略
            # 如果需要精确调整，可以应用旋转矩阵

        return image, mask, gaze_info_aug

  
    def _extract_eye_texture_from_reference(self, ref_image, ref_parsing, eye_size=256):
        """
        从正视参考图片中提取左右眼的正方形区域并拼接为256×512图像

        策略：
        1. 检测左右眼（标签2和5）
        2. 区分左右眼（按x坐标排序）
        3. 对每只眼睛：
           - 计算正方形边界框（基于外接圆半径）
           - 裁剪正方形矩形区域（保留完整区域，不应用掩码）
           - Resize到256×256
        4. 左右眼并排拼接：左眼(0:256) | 右眼(256:512)

        Args:
            ref_image: PIL.Image - 参考图像 (256x256)
            ref_parsing: PIL.Image or np.ndarray - 参考图像的人脸解析图
            eye_size: int - 每只眼睛的正方形尺寸 (默认256)

        Returns:
            eye_texture: PIL.Image - 左右眼拼接纹理 (256 x 512)
                         左眼在左半部分(0:256)，右眼在右半部分(256:512)
                         每只眼睛都是正方形矩形区域 (256×256)
            eye_regions_bbox: list - 眼睛区域bbox（归一化到[0, 1]）
                         格式: [left_x1, left_y1, left_x2, left_y2, right_x1, right_y1, right_x2, right_y2]
                         如果某只眼睛缺失，对应的4个值为空列表
        """
        # 转换为numpy array
        if isinstance(ref_image, Image.Image):
            ref_image_array = np.array(ref_image)
        else:
            ref_image_array = ref_image

        if isinstance(ref_parsing, Image.Image):
            parsing_array = np.array(ref_parsing, dtype=np.uint8)
        else:
            parsing_array = ref_parsing.astype(np.uint8)

        h, w = parsing_array.shape[:2]

        # 提取眼睛区域（标签2和5）
        eye_region = ((parsing_array == 2) | (parsing_array == 5)).astype(np.uint8)

        # 找到眼睛区域的轮廓
        contours, _ = cv2.findContours(eye_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 用于存储bbox信息
        eye_regions_bbox = []

        if len(contours) == 0:
            # 回退：返回整张脸的左右复制 (256 x 512)
            face_resized = cv2.resize(ref_image_array, (eye_size, eye_size), interpolation=cv2.INTER_LANCZOS4)
            output_canvas = np.zeros((eye_size, eye_size * 2, 3), dtype=np.uint8)
            output_canvas[:, 0:eye_size, :] = face_resized
            output_canvas[:, eye_size:eye_size*2, :] = face_resized
            return Image.fromarray(output_canvas), eye_regions_bbox

        # 收集所有眼睛轮廓的信息
        eye_info = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # 过滤小区域
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)

                eye_info.append({
                    'center': (int(center_x), int(center_y)),
                    'radius': radius,
                    'area': area,
                    'contour': contour
                })

        if len(eye_info) == 0:
            # 回退：返回整张脸的左右复制
            face_resized = cv2.resize(ref_image_array, (eye_size, eye_size), interpolation=cv2.INTER_LANCZOS4)
            output_canvas = np.zeros((eye_size, eye_size * 2, 3), dtype=np.uint8)
            output_canvas[:, 0:eye_size, :] = face_resized
            output_canvas[:, eye_size:eye_size*2, :] = face_resized
            return Image.fromarray(output_canvas), eye_regions_bbox

        # 按面积排序，取最大的两个（左右眼）
        eye_info = sorted(eye_info, key=lambda x: x['area'], reverse=True)[:2]

        # 按x坐标排序（从左到右）
        eye_info = sorted(eye_info, key=lambda x: x['center'][0])

        # 创建输出画布 (256 x 512)
        output_canvas = np.zeros((eye_size, eye_size * 2, 3), dtype=np.uint8)

        # 处理每只眼睛（左眼 -> 左半部分，右眼 -> 右半部分）
        for idx, eye in enumerate(eye_info[:2]):  # 最多处理2只眼睛
            center = eye['center']
            radius = eye['radius']
            contour = eye['contour']

            # 计算正方形边界框（基于扩展后的半径）
            expanded_radius = radius * self.eye_scale_factor
            side_length = int(expanded_radius * 2)  # 正方形边长

            center_x, center_y = center
            half_side = side_length // 2

            # 计算正方形边界
            x1 = max(0, center_x - half_side)
            y1 = max(0, center_y - half_side)
            x2 = min(w, center_x + half_side)
            y2 = min(h, center_y + half_side)

            # 确保是正方形（取最小边）
            crop_h = y2 - y1
            crop_w = x2 - x1
            crop_size = min(crop_h, crop_w)

            # 重新计算正方形边界（居中）
            x1 = max(0, center_x - crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)

            # 存储归一化的 bbox 坐标（像素坐标 -> [0, 1]）
            eye_regions_bbox.extend([x1 / w, y1 / h, x2 / w, y2 / h])

            # 裁剪正方形眼睛区域（保留完整矩形，不应用掩码）
            eye_crop = ref_image_array[y1:y2, x1:x2].copy()

            # Resize到256×256（保持正方形）
            if eye_crop.size > 0:
                eye_crop_resized = cv2.resize(eye_crop, (eye_size, eye_size), interpolation=cv2.INTER_LANCZOS4)

                # 放置到对应位置
                if idx == 0:
                    # 左眼 -> 左半部分 (0:256)
                    output_canvas[:, 0:eye_size, :] = eye_crop_resized
                elif idx == 1:
                    # 右眼 -> 右半部分 (256:512)
                    output_canvas[:, eye_size:eye_size*2, :] = eye_crop_resized

        # 如果只检测到一只眼睛，将其镜像复制
        if len(eye_info) == 1:
            output_canvas[:, eye_size:eye_size*2, :] = output_canvas[:, 0:eye_size, :]

        eye_texture = Image.fromarray(output_canvas)
        return eye_texture, eye_regions_bbox


    def __len__(self):
         return len(self.indices)


    def __getitem__(self, idx):
        """获取数据样本"""
        data_idx = self.indices[idx]

        # 使用线程本地 LMDB 环境
        env = get_thread_local_env(self.lmdb_path)

        with env.begin(write=False) as txn:
            # 从缓存中直接获取键名和视线信息
            image_key = self._image_keys_cache.get(data_idx)
            if image_key is None:
                raise ValueError(f"No image data found for index {data_idx}")

            # 直接读取图像数据
            image_data = txn.get(image_key.encode('utf-8'))
            if image_data is None:
                raise ValueError(f"No image data found for key {image_key}")

            # 从缓存获取视线信息
            gaze_info = self._gaze_info_cache.get(data_idx, {
                'gaze_x': 0.0, 'gaze_y': 0.0, 'head_x': 0.0, 'head_y': 0.0
            })

            # 解码图像
            image = Image.open(BytesIO(image_data)).convert('RGB')
            if image.size[0] != self.resolution:
                image = image.resize(
                    (self.resolution, self.resolution), resample=Image.LANCZOS)

            # 处理掩码 - 优先使用预计算掩码
            mask = None

            # 尝试读取预计算的眼部掩码
            if self.has_precomputed_masks:
                precomputed_mask_data = self._get_precomputed_eye_mask(
                    txn, data_idx)
                if precomputed_mask_data is not None:
                    mask_pil = Image.open(
                            BytesIO(precomputed_mask_data)).convert('L')
                    mask = np.array(mask_pil, dtype=np.uint8)
                        
                    # 根据配置处理掩码
                    if self.convert_to_square:
                    # 转换为正方形掩码（已包含 eye_scale_factor）
                        mask = self._convert_ellipse_mask_to_square(mask)
                        # print("已启用正方形掩码")
                    elif self.eye_scale_factor != 1.0:
                    # 保持椭圆形状，但应用扩张倍率
                        mask = self._expand_ellipse_mask(mask)

            # 应用数据增强（如果启用）
            if self.aug:
                image, mask, gaze_info = self._apply_augmentation(image, mask, gaze_info)

            # 图像变换
            image_tensor = self.transform(image)
            # Use ascontiguousarray to ensure contiguous memory layout
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)) / 255.0

            sample = {
                "image": image_tensor,
                "mask": mask_tensor,
                "norm_gaze": torch.tensor([gaze_info['gaze_x'], gaze_info['gaze_y']], dtype=torch.float32),
                "norm_head": torch.tensor([gaze_info['head_x'], gaze_info['head_y']], dtype=torch.float32),
                "gaze_x": gaze_info['gaze_x'],
                "gaze_y": gaze_info['gaze_y'],
                "head_x": gaze_info['head_x'],
                "head_y": gaze_info['head_y'],
                "subject_id": self._extract_subject_id_from_key(image_key) or "unknown"
            }
            if self.return_reference:
                subject_id = self._extract_subject_id_from_key(image_key)
                if subject_id and subject_id in self._subject_to_indices:
                    # 获取同一人物的所有数据索引（排除当前样本）
                    same_subject_indices = [i for i in self._subject_to_indices[subject_id] if i != data_idx]

                    if len(same_subject_indices) > 0:
                        # 选择最接近正视的reference（头部和注视都朝向正面）
                        best_ref_idx = None
                        min_angle_sum = float('inf')

                        for ref_idx in same_subject_indices:
                            ref_gaze_info = self._gaze_info_cache.get(ref_idx)
                            if ref_gaze_info:
                                # 计算gaze和head的角度偏离（越小越接近正视）
                                gaze_angle = abs(ref_gaze_info['gaze_x']) + abs(ref_gaze_info['gaze_y'])
                                head_angle = abs(ref_gaze_info['head_x']) + abs(ref_gaze_info['head_y'])
                                angle_sum = gaze_angle + head_angle

                                if angle_sum < min_angle_sum:
                                    min_angle_sum = angle_sum
                                    best_ref_idx = ref_idx

                        # 如果找到了正视的reference，使用它；否则随机选择
                        ref_data_idx = best_ref_idx if best_ref_idx is not None else random.choice(same_subject_indices)
                        ref_image_key = self._image_keys_cache.get(ref_data_idx)

                        if ref_image_key:
                            # 读取reference图像
                            ref_image_data = txn.get(ref_image_key.encode('utf-8'))
                            if ref_image_data is not None:
                                ref_image = Image.open(BytesIO(ref_image_data)).convert('RGB')
                                if ref_image.size[0] != self.resolution:
                                    ref_image = ref_image.resize(
                                        (self.resolution, self.resolution), resample=Image.LANCZOS)

                                # 将reference图像也进行transform
                                ref_image_tensor = self.transform(ref_image)
                                sample["reference_image"] = ref_image_tensor

                                # 读取reference的人脸解析图并提取眼睛纹理
                                ref_label_key = ref_image_key.replace('image-', 'label-')
                                ref_label_data = txn.get(ref_label_key.encode('utf-8'))

                                if ref_label_data is not None:
                                    ref_parsing_pil = Image.open(BytesIO(ref_label_data)).convert('L')
                                    if ref_parsing_pil.size[0] != self.resolution:
                                        ref_parsing_pil = ref_parsing_pil.resize(
                                            (self.resolution, self.resolution), resample=Image.NEAREST)

                                    # 提取眼睛纹理区域 (256x512: 左眼256×256 | 右眼256×256)
                                    eye_texture_pil, eye_bbox = self._extract_eye_texture_from_reference(
                                        ref_image, ref_parsing_pil, 256)
                                    # 转换为tensor
                                    eye_texture_tensor = self.transform(eye_texture_pil)
                                    sample["eye_reference"] = eye_texture_tensor

                                    # 添加眼睛 bbox（归一化到 [0, 1]）
                                    # 与 GazeCapture 保持一致：直接转换，不做默认值填充
                                    # - 检测到2只眼睛: [8] (left_x1, left_y1, left_x2, left_y2, right_x1, right_y1, right_x2, right_y2)
                                    # - 检测到1只眼睛: [4] (x1, y1, x2, y2)
                                    # - 没有检测到眼睛: [0] (空tensor)
                                    sample["eye_bbox"] = torch.tensor(eye_bbox, dtype=torch.float32)
    

            return sample

    def get_sample_info(self, idx):
        """获取样本信息（用于调试）"""
        data_idx = self.indices[idx]

        # 检查是否有预计算掩码
        has_precomputed_mask = data_idx in self._eyemask_keys_cache

        return {
            'data_idx': data_idx,
            'image_key': self._image_keys_cache.get(data_idx),
            'eyemask_key': self._eyemask_keys_cache.get(data_idx),
            'gaze_info': self._gaze_info_cache.get(data_idx),
            'has_precomputed_mask': has_precomputed_mask,
            'use_precomputed_masks': self.use_precomputed_masks,
            'has_precomputed_masks_available': self.has_precomputed_masks
        }

    def visualize_mask_on_image(self, idx, save_path=None, alpha=0.4):
        """
        可视化掩码叠加在原图上
        Args:
            idx: 样本索引
            save_path: 保存路径（可选）
            alpha: 掩码透明度
        """
        data = self[idx]
        image = data['image']
        mask = data['mask']
        # 反归一化
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        mask_np = mask.cpu().numpy()
        if mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = mask_np.astype(np.uint8)
        # 生成红色掩码
        mask_rgb = np.zeros_like(image_np)
        mask_rgb[..., 0] = mask_np  # 红色通道
        # 叠加
        overlay = image_np.copy()
        overlay = (1 - alpha) * overlay + alpha * mask_rgb
        overlay = np.clip(overlay, 0, 1)
        # 绘制
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image_np)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title('Mask Overlay')
        plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'✅ 可视化已保存到: {save_path}')
        else:
            plt.show()
        plt.close()


def draw_gaze_on_image(image, pitchyaw, thickness=2, color=(0, 255, 255)):
    """在图像上绘制视线方向箭头"""
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

    image_out = image.copy()
    (h, w) = image.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 2.0))

    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])

    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)),
                    color, thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out


def debug_lmdb_structure(lmdb_path, max_keys=20):
    """调试LMDB数据库结构"""
    env = lmdb.open(lmdb_path, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        keys = []
        count = 0
        for key, value in cursor:
            if count >= max_keys:
                break
            key_str = key.decode('utf-8')
            keys.append(key_str)
            count += 1
    env.close()
    return keys


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import sys

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if len(sys.argv) > 1:
        lmdb_path = sys.argv[1]
    else:
        lmdb_path = "/mnt/data/xhy/GC-GAN/ETH-GAZE/Vae-Gaze/dataset_lmdb"

    if os.path.exists(lmdb_path):
        print("=" * 60)
        print("测试1: 使用原始椭圆掩码 (无扩张)")
        print("=" * 60)
        dataset_original = ETHXGazeLMDBDataset(
            lmdb_path=lmdb_path,
            resolution=256,
            aug=False,
            split="val",
            eye_scale_factor=1.0,  # 不扩张
            use_precomputed_masks=True,
            convert_to_square=False
        )

        dataloader = DataLoader(dataset_original, batch_size=2, shuffle=True)
        batch = next(iter(dataloader))

        print("✅ 原始椭圆掩码测试成功!")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        print(f"🔵 掩码类型: 椭圆形 (原始大小)")
        print(f"📏 扩张倍率: 1.0x")
        
        print("\n" + "=" * 60)
        print("测试2: 使用扩张的椭圆掩码")
        print("=" * 60)
        dataset_ellipse = ETHXGazeLMDBDataset(
            lmdb_path=lmdb_path,
            resolution=256,
            aug=False,
            split="val",
            eye_scale_factor=1.2,  # 扩张1.2倍
            use_precomputed_masks=True,
            convert_to_square=False
        )

        dataloader = DataLoader(dataset_ellipse, batch_size=2, shuffle=True)
        batch = next(iter(dataloader))

        print("✅ 扩张椭圆掩码测试成功!")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        print(f"🔵 掩码类型: 椭圆形 (扩张)")
        print(f"📏 扩张倍率: 1.2x")
        
        print("\n" + "=" * 60)
        print("测试3: 将椭圆掩码转换为正方形掩码")
        print("=" * 60)
        dataset_square = ETHXGazeLMDBDataset(
            lmdb_path=lmdb_path,
            resolution=256,
            aug=False,
            split="val",
            eye_scale_factor=1.2,
            use_precomputed_masks=True,
            convert_to_square=True  # 转换为正方形掩码
        )

        dataloader = DataLoader(dataset_square, batch_size=2, shuffle=True)
        batch = next(iter(dataloader))

        print("✅ 正方形掩码测试成功!")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        print(f"🟦 掩码类型: 正方形")
        print(f"📏 扩张倍率: 1.2x")
        
        # 显示第一个样本的信息
        print("\n" + "=" * 60)
        print("样本信息:")
        print("=" * 60)
        sample_info = dataset_ellipse.get_sample_info(0)
        print(f"📊 {sample_info}")
    else:
        print(f"❌ LMDB路径不存在: {lmdb_path}")
