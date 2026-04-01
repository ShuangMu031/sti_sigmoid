import os
import cv2
import numpy as np
import time
import pickle
from pathlib import Path
from dataclasses import dataclass

from stitcher.common.logger import get_logger
from stitcher.common.image_utils import ensure_directory
from stitcher.io.image_io import cv_imread, cv_imwrite
from stitcher.config import SEAM_BAND, FEATHER_RADIUS
from stitcher.algorithms import (
    mbs_saliency,
    canny_edge_detect,
    registerTexture,
    homography_align,
    graph_cut_seam,
    gradient_blend_local
)

logger = get_logger(__name__)

# 缓存文件路径
CACHE_DIR = Path(__file__).parent.parent.parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class PairCacheEntry:
    img1_w: np.ndarray
    img2_w: np.ndarray
    valid1: np.ndarray
    valid2: np.ndarray
    overlap: np.ndarray
    label_map: np.ndarray
    base: np.ndarray
    seam_line: np.ndarray


class PairCache:
    def __init__(self):
        self.cache = {}
        self._load_from_disk()
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, entry):
        self.cache[key] = entry
        self._save_to_disk()
    
    def clear(self):
        self.cache.clear()
        self._save_to_disk()
    
    def _get_cache_file(self):
        return CACHE_DIR / 'pair_cache.pkl'
    
    def _save_to_disk(self):
        try:
            # 只保存关键信息，不保存完整图像数据
            # 注意：这里只是示例，实际中需要更复杂的序列化策略
            # 由于图像数据较大，不建议直接序列化到文件
            # 这里仅作为演示
            pass
        except Exception as e:
            logger.warning(f"保存缓存到磁盘失败: {e}")
    
    def _load_from_disk(self):
        try:
            # 从磁盘加载缓存
            # 注意：这里只是示例，实际中需要更复杂的反序列化策略
            pass
        except Exception as e:
            logger.warning(f"从磁盘加载缓存失败: {e}")


def compute_image_hash(img):
    if img is None:
        return 0
    # 使用更高效的哈希方法：计算图像的平均像素值和形状作为哈希
    # 对于大图，只采样中心区域
    h, w = img.shape[:2]
    # 采样中心 100x100 区域
    sample_h = min(100, h)
    sample_w = min(100, w)
    start_h = (h - sample_h) // 2
    start_w = (w - sample_w) // 2
    sample = img[start_h:start_h+sample_h, start_w:start_w+sample_w]
    # 计算各通道的平均值
    mean_values = np.mean(sample, axis=(0, 1))
    # 结合形状和均值生成哈希
    hash_value = hash((h, w, tuple(mean_values.tolist())))
    return hash_value


class StitchingPipeline:
    def __init__(self):
        self.logger = logger
        self.image_paths = []
        self.config = {
            'SEAM_BAND': SEAM_BAND,
            'FEATHER_RADIUS': FEATHER_RADIUS
        }
        self.result_image = None
        self.progress_callback = None
        self.pair_cache = PairCache()

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def _report_progress(self, step, total, message):
        if self.progress_callback:
            self.progress_callback(step, total, message)

    def load_images(self, image_paths):
        self.image_paths = image_paths
        return True

    def update_config(self, new_config):
        self.config.update(new_config)

    def _extract_seam_line(self, label_map, overlap):
        seam = np.zeros_like(label_map, np.uint8)
        seam[:, 1:] |= (label_map[:, 1:] != label_map[:, :-1])
        seam[1:, :] |= (label_map[1:, :] != label_map[:-1, :])
        seam &= overlap.astype(np.uint8)
        return seam

    def _prepare_pair(self, moving_img, base_img, pair_prefix, global_step_offset, total_pairs):
        self._report_progress(global_step_offset + 1, max(total_pairs * 7, 1), f"{pair_prefix}：执行显著性和边缘检测...")
        sal1 = mbs_saliency(moving_img)
        sal2 = mbs_saliency(base_img)
        edge1, _ = canny_edge_detect(moving_img)
        edge2, _ = canny_edge_detect(base_img)

        self._report_progress(global_step_offset + 2, max(total_pairs * 7, 1), f"{pair_prefix}：计算单应性变换矩阵...")
        h_matrix = registerTexture(moving_img, edge1, base_img, edge2)
        self.logger.info(f"{pair_prefix} 单应性变换矩阵: {h_matrix}")
        
        return sal1, sal2, edge1, edge2, h_matrix

    def _align_pair(self, moving_img, base_img, sal1, sal2, edge1, edge2, h_matrix, pair_prefix, global_step_offset, total_pairs):
        self._report_progress(global_step_offset + 3, max(total_pairs * 7, 1), f"{pair_prefix}：图像对齐中...")
        img1_w, img2_w, sal1_w, sal2_w, edge1_w, edge2_w, valid1, valid2 = homography_align(
            moving_img, sal1, edge1,
            base_img, sal2, edge2,
            h_matrix
        )

        overlap = valid1 & valid2
        return img1_w, img2_w, sal1_w, sal2_w, edge1_w, edge2_w, valid1, valid2, overlap

    def _cut_pair(self, img1_w, img2_w, sal1_w, sal2_w, edge1_w, edge2_w, valid1, valid2, overlap, pair_prefix, global_step_offset, total_pairs):
        self._report_progress(global_step_offset + 4, max(total_pairs * 7, 1), f"{pair_prefix}：执行图割接缝选择...")
        label_map = graph_cut_seam(
            img1_w, img2_w,
            sal1_w, sal2_w,
            edge1_w, edge2_w,
            valid1, valid2,
            overlap
        )

        self._report_progress(global_step_offset + 5, max(total_pairs * 7, 1), f"{pair_prefix}：执行硬合成...")
        base = img2_w.copy()
        base[label_map == 1] = img1_w[label_map == 1]

        seam_line = self._extract_seam_line(label_map, overlap)
        return label_map, base, seam_line

    def _blend_pair(self, img1_w, base, seam_line, overlap, seam_band_size, feather_radius, pair_prefix, global_step_offset, total_pairs):
        self._report_progress(global_step_offset + 6, max(total_pairs * 7, 1), f"{pair_prefix}：执行局部梯度融合...")
        kernel = np.ones((seam_band_size, seam_band_size), np.uint8)
        seam_band = cv2.dilate(seam_line.astype(np.uint8), kernel)
        seam_band = cv2.erode(seam_band, np.ones((3, 3), np.uint8), iterations=2)
        poisson_mask = seam_band.astype(bool) & overlap

        result = gradient_blend_local(img1_w, base, poisson_mask)

        self._report_progress(global_step_offset + 7, max(total_pairs * 7, 1), f"{pair_prefix}：执行羽化平滑...")
        alpha = cv2.GaussianBlur(
            seam_band.astype(np.float32),
            (0, 0),
            sigmaX=feather_radius
        )
        alpha = alpha / (alpha.max() + 1e-6)
        alpha = np.clip(alpha, 0, 1)[..., None]

        result = (1 - alpha) * result + alpha * base
        return result.astype(np.uint8)

    def _stitch_pair(self, moving_img, base_img, pair_index=1, total_pairs=1, global_step_offset=0):
        pair_prefix = f"第 {pair_index}/{total_pairs} 组"
        seam_band_size = self.config.get('SEAM_BAND', 9)
        feather_radius = self.config.get('FEATHER_RADIUS', 11)
        timing_info = {}

        # 计算缓存键
        moving_hash = compute_image_hash(moving_img)
        base_hash = compute_image_hash(base_img)
        cache_key = (moving_hash, base_hash, pair_index)

        # 尝试从缓存获取
        cache_entry = self.pair_cache.get(cache_key)
        if cache_entry:
            self.logger.info(f"{pair_prefix}：从缓存加载中间结果")
            img1_w, img2_w, valid1, valid2, overlap, label_map, base, seam_line = (
                cache_entry.img1_w, cache_entry.img2_w, cache_entry.valid1, cache_entry.valid2,
                cache_entry.overlap, cache_entry.label_map, cache_entry.base, cache_entry.seam_line
            )
        else:
            # 执行完整流程
            start_time = time.time()
            sal1, sal2, edge1, edge2, h_matrix = self._prepare_pair(
                moving_img, base_img, pair_prefix, global_step_offset, total_pairs
            )
            timing_info['prepare'] = time.time() - start_time
            
            start_time = time.time()
            img1_w, img2_w, sal1_w, sal2_w, edge1_w, edge2_w, valid1, valid2, overlap = self._align_pair(
                moving_img, base_img, sal1, sal2, edge1, edge2, h_matrix, pair_prefix, global_step_offset, total_pairs
            )
            timing_info['align'] = time.time() - start_time
            
            start_time = time.time()
            label_map, base, seam_line = self._cut_pair(
                img1_w, img2_w, sal1_w, sal2_w, edge1_w, edge2_w, valid1, valid2, overlap, pair_prefix, global_step_offset, total_pairs
            )
            timing_info['cut'] = time.time() - start_time
            
            # 缓存中间结果
            cache_entry = PairCacheEntry(
                img1_w=img1_w,
                img2_w=img2_w,
                valid1=valid1,
                valid2=valid2,
                overlap=overlap,
                label_map=label_map,
                base=base,
                seam_line=seam_line
            )
            self.pair_cache.set(cache_key, cache_entry)

        # 执行融合阶段（每次都需要重新执行，因为参数可能变化）
        start_time = time.time()
        result = self._blend_pair(
            img1_w, base, seam_line, overlap, seam_band_size, feather_radius, pair_prefix, global_step_offset, total_pairs
        )
        timing_info['blend'] = time.time() - start_time
        
        self.logger.info(f"{pair_prefix} 阶段耗时: {timing_info}")
        return result

    def run(self, output_path=None, preview_path=None):
        try:
            t0 = time.time()
            timing_info = {}

            if output_path:
                ensure_directory(str(Path(output_path).parent))
            else:
                ensure_directory("./outputs")

            if len(self.image_paths) < 2:
                self.logger.error("需要至少两张图像进行拼接")
                return False

            # 流式读取：只保留当前结果和下一张图
            self._report_progress(0, 1, "正在加载第一张图像...")
            result = cv_imread(self.image_paths[0])
            if result is None:
                self.logger.error(f"无法读取图像文件: {self.image_paths[0]}")
                return False

            total_pairs = len(self.image_paths) - 1
            total_steps = max(total_pairs * 7, 1)
            self._report_progress(0, total_steps, f"开始多图拼接，共 {len(self.image_paths)} 张图像...")

            for pair_idx in range(1, len(self.image_paths)):
                self._report_progress(0, total_steps, f"正在加载第 {pair_idx + 1} 张图像...")
                base_img = cv_imread(self.image_paths[pair_idx])
                if base_img is None:
                    self.logger.error(f"无法读取图像文件: {self.image_paths[pair_idx]}")
                    return False

                moving_img = result
                result = self._stitch_pair(
                    moving_img,
                    base_img,
                    pair_index=pair_idx,
                    total_pairs=total_pairs,
                    global_step_offset=(pair_idx - 1) * 7
                )

                # 释放内存
                del base_img
                del moving_img

            self.result_image = result

            if output_path:
                save_start = time.time()
                self._report_progress(total_steps, total_steps, "正在保存结果...")
                if not cv_imwrite(output_path, self.result_image):
                    raise RuntimeError(f"无法保存结果图像: {output_path}")
                timing_info['save'] = time.time() - save_start

            # 生成预览图
            if preview_path:
                preview_start = time.time()
                self._report_progress(total_steps, total_steps, "正在生成预览图...")
                # 生成缩略预览图
                preview_img = self._generate_preview(self.result_image)
                if not cv_imwrite(preview_path, preview_img):
                    raise RuntimeError(f"无法保存预览图像: {preview_path}")
                timing_info['preview'] = time.time() - preview_start

            total_time = time.time() - t0
            self.logger.info(f"拼接完成！总耗时: {total_time:.2f}s")
            self.logger.info(f"阶段耗时: {timing_info}")
            self._report_progress(total_steps, total_steps, "拼接完成!")
            return self.result_image

        except Exception as e:
            self.logger.error(f"拼接过程中出错: {str(e)}")
            return False

    def _generate_preview(self, image, max_size=1000):
        """生成预览图"""
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image
        
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def save_result(self, output_path):
        if self.result_image is not None:
            return cv_imwrite(output_path, self.result_image)
        return False
