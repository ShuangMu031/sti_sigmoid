import cv2
import numpy as np
from scipy.ndimage import label  # 兼容连通域标记

# ===================== MATLAB 函数等效实现 =====================
def bwareaopen(binary_img, min_area):
    """
    复刻 MATLAB 的 bwareaopen 函数：去除小面积连通域
    输入：
        binary_img - bool 型二值图（True=1，False=0）
        min_area   - 最小保留面积（像素数）
    输出：
        filtered_img - 处理后的 bool 型二值图
    """
    # 转换为 uint8 格式（OpenCV 要求）
    img_uint8 = (binary_img * 255).astype(np.uint8)
    # 查找连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_uint8, connectivity=8)
    # 保留面积 >= min_area 的区域
    filtered_img = np.zeros_like(binary_img)
    for i in range(1, num_labels):  # 跳过背景（0号区域）
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_img[labels == i] = True
    return filtered_img

def imfill(binary_img):
    """
    复刻 MATLAB 的 imfill(binary_img, 'holes')：填充二值图中的孔洞
    输入：binary_img - bool 型二值图
    输出：filled_img - 填充后的 bool 型二值图
    """
    img_uint8 = (binary_img * 255).astype(np.uint8)
    # 复制图像用于填充
    img_floodfill = img_uint8.copy()
    # 定义填充的掩膜（比原图大2像素，避免边界问题）
    h, w = img_uint8.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # 从左上角开始泛洪填充（背景）
    cv2.floodFill(img_floodfill, mask, (0, 0), 255)
    # 反转填充结果，得到孔洞区域
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    # 合并原图和孔洞区域（填充孔洞）
    filled_uint8 = img_uint8 | img_floodfill_inv
    return filled_uint8 > 0

# ===================== 核心函数（复刻 detect_object_height.m） =====================
def detect_object_height(img, pmap, edge_mask):
    """
    完整复刻 MATLAB 的 detect_object_height.m 函数
    功能：检测图像中主要物体的像素高度（垂直方向像素数）
    输入：
        img         - 输入图像（np.float64 类型，0-1 范围）
        pmap        - 显著性图（与 img 同尺寸，0-1 范围）
        edge_mask   - 边缘掩码（bool 型，True=边缘，False=非边缘）
    输出：
        obj_height  - 主要物体的像素高度（int，垂直方向最大跨度）
        obj_mask    - 主要物体的二值掩码（bool 型，True=物体，False=背景）
    """
    # ========== MATLAB 步骤1：预处理 - 结合显著性和边缘提取物体候选区域 ==========
    # 1.1 显著性图阈值分割（pmap > 0.3）
    pmap_thresh = pmap > 0.3

    # 1.2 复刻：edge_mask | ~bwareaopen(~edge_mask, 400)
    # ~edge_mask → 非边缘区域；bwareaopen去除面积<400的非边缘区域；~取反 → 保留大面积非边缘背景
    non_edge = ~edge_mask
    non_edge_filtered = bwareaopen(non_edge, 400)
    edge_combined = edge_mask | ~non_edge_filtered
    # 与显著性区域相交，得到物体候选区域
    obj_candidate = pmap_thresh & edge_combined

    # 1.3 形态学处理：imclose + imfill（复刻 MATLAB 逻辑）
    # 创建圆形结构元素（strel('disk', 3) → 直径7的椭圆核）
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 半径3 → 直径7
    # 转换为 uint8 格式进行形态学操作
    obj_candidate_uint8 = (obj_candidate * 255).astype(np.uint8)
    # imclose：闭运算（先膨胀后腐蚀，闭合小缝隙）
    obj_closed_uint8 = cv2.morphologyEx(obj_candidate_uint8, cv2.MORPH_CLOSE, se)
    obj_closed = obj_closed_uint8 > 0
    # imfill：填充内部孔洞
    obj_candidate = imfill(obj_closed)

    # ========== MATLAB 步骤2：提取最大连通区域 ==========
    # 2.1 连通域标记（复刻 bwlabel）
    # label 函数返回 (标记矩阵, 区域数量)，与 MATLAB bwlabel 一致
    L, num = label(obj_candidate, structure=np.ones((3, 3)))  # 8连通域（与MATLAB默认一致）
    if num == 0:
        # 无物体时返回0和空掩码
        obj_height = 0
        obj_mask = np.zeros(img.shape[:2], dtype=bool)
        return obj_height, obj_mask

    # 2.2 计算每个区域的面积，选最大区域
    area = np.zeros(num, dtype=int)
    for i in range(1, num + 1):  # MATLAB 索引从1开始，Python label 也从1开始
        area[i - 1] = np.sum(L == i)  # 统计第i个区域的像素数
    max_idx = np.argmax(area) + 1  # 最大区域的索引（转回1开始）
    obj_mask = L == max_idx  # 主要物体的二值掩码

    # ========== MATLAB 步骤3：计算物体高度 ==========
    # 3.1 找到物体区域的所有行坐标（复刻 find(obj_mask)）
    row, _ = np.where(obj_mask)
    if len(row) == 0:
        obj_height = 0
        return obj_height, obj_mask

    # 3.2 高度 = 最大行索引 - 最小行索引 + 1（包含首尾像素）
    min_row = np.min(row)
    max_row = np.max(row)
    obj_height = max_row - min_row + 1

    # ========== MATLAB 步骤4：可选可视化（注释保留，与原逻辑一致） ==========
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plt.figure('Object Height Detection')
    # 子图1：原始图像
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    # 子图2：标记物体区域
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # 找到物体的列范围
    col, _ = np.where(obj_mask)
    min_col = np.min(col)
    max_col = np.max(col)
    # 绘制红色矩形框
    rect = patches.Rectangle(
        (min_col, min_row),  # 左上角坐标
        max_col - min_col + 1,  # 宽度
        obj_height,  # 高度
        linewidth=4,
        edgecolor='r',
        facecolor='none'
    )
    plt.gca().add_patch(rect)
    plt.title(f'Object Height: {obj_height} pixels')
    plt.axis('off')
    plt.show()
    """

    return obj_height, obj_mask