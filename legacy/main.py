# =========================================================
# main_final_matlab_style.py
# GraphCut + Hard Composite + Local Gradient Blend
# 图像拼接主程序 - 实现基于图割和局部梯度融合的无缝图像拼接
# =========================================================

import os
import cv2
import numpy as np
import traceback
import time

# 导入各功能模块
from edge_detect import canny_edge_detect  # 边缘检测模块
from homography_align import homography_align  # 同态变换对齐模块
from register_texture import registerTexture  # 纹理配准模块
from saliency import mbs_saliency  # 显著性检测模块
from overlap import compute_overlap_masks  # 重叠区域计算模块
from graph_cut import graph_cut_seam  # 图割接缝选择模块
from gradient_blend import gradient_blend_local  # 局部梯度融合模块


def cv_imread(file_path):
    """支持中文路径的图像读取函数"""
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def cv_imwrite(file_path, img):
    """支持中文路径的图像保存函数"""
    ext = os.path.splitext(file_path)[1]
    cv2.imencode(ext, img)[1].tofile(file_path)


# ---------------------------------------------------------
# 接缝线提取函数 (仅用于可视化)
# ---------------------------------------------------------
def extract_seam_line(label_map, overlap):
    """
    从标签图中提取接缝线位置
    
    参数:
        label_map: 图割生成的标签图(0表示属于图像2，1表示属于图像1)
        overlap: 重叠区域掩码
    
    返回:
        seam: 1像素宽的接缝线掩码
    """
    # 创建与标签图相同大小的零矩阵
    seam = np.zeros_like(label_map, np.uint8)
    # 检测水平方向的标签变化（左右相邻像素不同）
    seam[:, 1:] |= (label_map[:, 1:] != label_map[:, :-1])
    # 检测垂直方向的标签变化（上下相邻像素不同）
    seam[1:, :] |= (label_map[1:, :] != label_map[:-1, :])
    # 只保留重叠区域内的接缝
    seam &= overlap.astype(np.uint8)
    return seam


# ---------------------------------------------------------
# 主函数
# ---------------------------------------------------------
def main():
    """
    图像拼接主流程函数
    实现步骤：图像加载 → 显著性和边缘检测 → 同态变换计算 → 图像对齐 → 
              图割接缝选择 → 硬合成 → 局部梯度融合 → 羽化平滑
    """

    try:
        t0 = time.time()  # 记录开始时间
        
        # 获取脚本所在目录，确保路径正确
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 创建输出目录
        os.makedirs(os.path.join(script_dir, "output"), exist_ok=True)

        # -------------------------------------------------
        # 1. 加载图像
        # -------------------------------------------------
        img_dir = os.path.join(script_dir, "Imgs")  # 图像目录
        # 获取目录中所有jpg和png图像文件
        imgs = sorted([f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg", ".png", ".JPG", ".PNG"))])

        # 读取两张待拼接图像
        img1 = cv_imread(os.path.join(img_dir, imgs[0]))
        img2 = cv_imread(os.path.join(img_dir, imgs[1]))

        # -------------------------------------------------
        # 2. 显著性和边缘检测
        # -------------------------------------------------
        # 计算图像显著性（用于图割权重）
        sal1 = mbs_saliency(img1)
        sal2 = mbs_saliency(img2)
        # 计算图像边缘（用于同态变换和图割）
        edge1, _ = canny_edge_detect(img1)
        edge2, _ = canny_edge_detect(img2)

        # -------------------------------------------------
        # 3. 计算同态变换矩阵
        # -------------------------------------------------
        # 根据图像纹理特征计算同态变换矩阵
        H = registerTexture(img1, edge1, img2, edge2)
        print("\nHomography:\n", H)  # 打印同态变换矩阵

        # -------------------------------------------------
        # 4. 将图像变换到公共画布
        # -------------------------------------------------
        # 应用同态变换，将两张图像对齐到同一坐标系
        img1_w, img2_w, sal1_w, sal2_w, edge1_w, edge2_w, valid1, valid2 = homography_align(
            img1, sal1, edge1,
            img2, sal2, edge2,
            H
        )

        # 计算重叠区域掩码
        _, _, overlap = compute_overlap_masks(img1_w, img2_w)

        # -------------------------------------------------
        # 5. 图割接缝选择（生成标签图）
        # -------------------------------------------------
        # 使用图割算法选择最优接缝位置
        label_map = graph_cut_seam(
            img1_w, img2_w,  # 变换后的图像
            sal1_w, sal2_w,  # 变换后的显著性图
            edge1_w, edge2_w,  # 变换后的边缘图
            valid1, valid2,  # 有效区域掩码
            overlap  # 重叠区域掩码
        )

        # -------------------------------------------------
        # 6. 硬合成（MATLAB 等价步骤）
        # -------------------------------------------------
        # 关键：GraphCut 给的是"区域归属"
        # 以图像2为基础，将属于图像1的区域替换为图像1的内容
        base = img2_w.copy()
        base[label_map == 1] = img1_w[label_map == 1]

        # -------------------------------------------------
        # 7. 接缝线可视化（已删除调试输出）
        # -------------------------------------------------
        seam_line = extract_seam_line(label_map, overlap)
        seam_vis = cv2.dilate(seam_line, np.ones((3, 3), np.uint8))

        # -------------------------------------------------
        # 8. 局部泊松融合（MATLAB风格的带状区域）
        # -------------------------------------------------
        SEAM_BAND = 9  # 融合带宽参数 ⭐ 先缩小（非常重要）

        # 膨胀接缝线，创建融合带宽
        kernel = np.ones((SEAM_BAND, SEAM_BAND), np.uint8)
        seam_band = cv2.dilate(seam_line.astype(np.uint8), kernel)

        # 🔑 关键 1：向内收缩，避免接触重叠区域/图像边界
        seam_band = cv2.erode(
            seam_band,
            np.ones((3, 3), np.uint8),
            iterations=2
        )

        # 确定需要进行泊松融合的区域（融合带宽与重叠区域的交集）
        poisson_mask = seam_band.astype(bool) & overlap

        # 执行局部梯度融合
        result = gradient_blend_local(
            source=img1_w,  # 源图像（图像1）
            target=base,  # 目标图像（硬合成结果）
            mask=poisson_mask  # 融合区域掩码
        )
        
        # -------------------------------------------------
        # 9. 接缝感知的羽化平滑（最终优化）
        # -------------------------------------------------
        # seam_line: 原始接缝
        # result   : 泊松融合输出
        # base     : 硬合成结果

        FEATHER_RADIUS = 11  # 羽化半径 ⭐ 推荐 7~11

        # 生成羽化权重图
        alpha = cv2.GaussianBlur(
            seam_band.astype(np.float32),
            (0, 0),
            sigmaX=FEATHER_RADIUS
        )

        # 归一化权重图
        alpha = alpha / (alpha.max() + 1e-6)
        alpha = np.clip(alpha, 0, 1)[..., None]  # 添加通道维度

        # 🔑 抹匀：向硬合成收敛
        result = (1 - alpha) * result + alpha * base
        result = result.astype(np.uint8)

        # 保存最终结果
        output_path = os.path.join(script_dir, "output", "result.png")
        cv_imwrite(output_path, result)

        # 打印完成信息
        print(f"[OK] 拼接完成！最终结果已保存至 {output_path}")
        print(f"[TIME] 总耗时: {time.time() - t0:.2f}s")

    except Exception:
        # 捕获并打印异常信息
        traceback.print_exc()


if __name__ == "__main__":
    main()  # 程序入口点