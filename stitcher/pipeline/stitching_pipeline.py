import os
import cv2
import numpy as np
import time
from pathlib import Path
import concurrent.futures

from stitcher.common.logger import get_logger
from stitcher.common.image_utils import ensure_directory
from stitcher.io.image_io import cv_imread, cv_imwrite
from stitcher.config import SEAM_BAND, FEATHER_RADIUS, FEATURE_DETECTOR
from stitcher.algorithms import (
    mbs_saliency,
    canny_edge_detect,
    registerTexture,
    homography_align,
    compute_overlap_masks,
    graph_cut_seam,
    gradient_blend_local,
    sort_images_by_overlap
)

logger = get_logger(__name__)


class StitchingPipeline:
    def __init__(self):
        self.logger = logger
        self.image_paths = []
        self.config = {
            'SEAM_BAND': SEAM_BAND,
            'FEATHER_RADIUS': FEATHER_RADIUS,
            'AUTO_SORT': True,
            'FEATURE_DETECTOR': FEATURE_DETECTOR
        }
        self.result_image = None
        self.progress_callback = None

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

    def _extract_features(self, img):
        """提取图像特征"""
        sal = mbs_saliency(img)
        edge, _ = canny_edge_detect(img)
        return sal, edge

    def _stitch_pair(self, moving_img, base_img, pair_index=1, total_pairs=1, global_step_offset=0):
        phase_total = 7
        pair_prefix = f"第 {pair_index}/{total_pairs} 组"

        seam_band_size = self.config.get('SEAM_BAND', 9)
        feather_radius = self.config.get('FEATHER_RADIUS', 11)

        self._report_progress(global_step_offset + 1, max(total_pairs * phase_total, 1), f"{pair_prefix}：执行显著性和边缘检测...")
        
        # 并行处理特征提取
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self._extract_features, moving_img)
            future2 = executor.submit(self._extract_features, base_img)
            sal1, edge1 = future1.result()
            sal2, edge2 = future2.result()

        self._report_progress(global_step_offset + 2, max(total_pairs * phase_total, 1), f"{pair_prefix}：计算单应性变换矩阵...")
        h_matrix = registerTexture(moving_img, edge1, base_img, edge2)
        self.logger.info(f"{pair_prefix} 单应性变换矩阵: {h_matrix}")

        self._report_progress(global_step_offset + 3, max(total_pairs * phase_total, 1), f"{pair_prefix}：图像对齐中...")
        img1_w, img2_w, sal1_w, sal2_w, edge1_w, edge2_w, valid1, valid2 = homography_align(
            moving_img, sal1, edge1,
            base_img, sal2, edge2,
            h_matrix
        )

        _, _, overlap = compute_overlap_masks(img1_w, img2_w)

        self._report_progress(global_step_offset + 4, max(total_pairs * phase_total, 1), f"{pair_prefix}：执行图割接缝选择...")
        label_map = graph_cut_seam(
            img1_w, img2_w,
            sal1_w, sal2_w,
            edge1_w, edge2_w,
            valid1, valid2,
            overlap
        )

        self._report_progress(global_step_offset + 5, max(total_pairs * phase_total, 1), f"{pair_prefix}：执行硬合成...")
        base = img2_w.copy()
        base[label_map == 1] = img1_w[label_map == 1]

        seam_line = self._extract_seam_line(label_map, overlap)

        self._report_progress(global_step_offset + 6, max(total_pairs * phase_total, 1), f"{pair_prefix}：执行局部梯度融合...")
        kernel = np.ones((seam_band_size, seam_band_size), np.uint8)
        seam_band = cv2.dilate(seam_line.astype(np.uint8), kernel)
        seam_band = cv2.erode(seam_band, np.ones((3, 3), np.uint8), iterations=2)
        poisson_mask = seam_band.astype(bool) & overlap

        result = gradient_blend_local(img1_w, base, poisson_mask)

        self._report_progress(global_step_offset + 7, max(total_pairs * phase_total, 1), f"{pair_prefix}：执行羽化平滑...")
        alpha = cv2.GaussianBlur(
            seam_band.astype(np.float32),
            (0, 0),
            sigmaX=feather_radius
        )
        alpha = alpha / (alpha.max() + 1e-6)
        alpha = np.clip(alpha, 0, 1)[..., None]

        result = (1 - alpha) * result + alpha * base
        return result.astype(np.uint8)

    def run(self, output_path=None):
        try:
            t0 = time.time()

            if output_path:
                ensure_directory(str(Path(output_path).parent))
            else:
                ensure_directory("./outputs")

            if len(self.image_paths) < 2:
                self.logger.error("需要至少两张图像进行拼接")
                return False

            self._report_progress(0, 1, "正在加载图像...")
            images = []
            for idx, image_path in enumerate(self.image_paths, start=1):
                img = cv_imread(image_path)
                if img is None:
                    self.logger.error(f"无法读取图像文件: {image_path}")
                    return False
                images.append(img)
                self._report_progress(0, 1, f"正在加载图像 {idx}/{len(self.image_paths)}...")

            if self.config.get('AUTO_SORT', True) and len(images) > 2:
                self._report_progress(0, 1, "正在分析图像顺序...")
                order = sort_images_by_overlap(
                    images, 
                    detector_type=self.config.get('FEATURE_DETECTOR', 'ORB')
                )
                images = [images[i] for i in order]
                self.logger.info(f"图像排序结果: {order}")

            result = images[0]
            total_pairs = len(images) - 1
            total_steps = max(total_pairs * 7, 1)
            self._report_progress(0, total_steps, f"开始多图拼接，共 {len(images)} 张图像...")

            for pair_idx in range(1, len(images)):
                moving_img = result
                base_img = images[pair_idx]
                result = self._stitch_pair(
                    moving_img,
                    base_img,
                    pair_index=pair_idx,
                    total_pairs=total_pairs,
                    global_step_offset=(pair_idx - 1) * 7
                )

            self.result_image = result

            if output_path:
                self._report_progress(total_steps, total_steps, "正在保存结果...")
                if not cv_imwrite(output_path, self.result_image):
                    raise RuntimeError(f"无法保存结果图像: {output_path}")

            self.logger.info(f"拼接完成！总耗时: {time.time() - t0:.2f}s")
            self._report_progress(total_steps, total_steps, "拼接完成!")
            return self.result_image

        except Exception as e:
            self.logger.error(f"拼接过程中出错: {str(e)}")
            return False

    def save_result(self, output_path):
        if self.result_image is not None:
            return cv_imwrite(output_path, self.result_image)
        return False
