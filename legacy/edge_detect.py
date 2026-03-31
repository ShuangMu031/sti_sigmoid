import cv2
import numpy as np
from config import CANNY_LOW_THRESH, CANNY_HIGH_THRESH, CANNY_SIGMA


def canny_edge_detect(img, low_thresh=None, high_thresh=None, sigma=None):
    """
    完整复刻 MATLAB 的 canny_edge_detect.m 函数
    功能：对输入图像执行 Canny 边缘检测，输出二值边缘掩码和可视化边缘图
    输入：
        img         - 输入图像（支持 np.float64(0-1)/uint8(0-255)，彩色(RGB)/灰度均可）
        low_thresh  - Canny 低阈值（支持0-255范围或0-1范围，默认使用 config 配置）
        high_thresh - Canny 高阈值（支持0-255范围或0-1范围，默认使用 config 配置）
        sigma       - 高斯平滑核标准差（默认使用 config 配置）
    输出：
        edge_mask   - 二值边缘掩码（bool 型，与 img 同尺寸，边缘=True，非边缘=False）
        edge_img    - 边缘可视化图（np.float64 型，0-1 范围，边缘=1，背景=0）
    """
    # ========== 参数默认值设置 ==========
    low_thresh = CANNY_LOW_THRESH if low_thresh is None else low_thresh
    high_thresh = CANNY_HIGH_THRESH if high_thresh is None else high_thresh
    sigma = CANNY_SIGMA if sigma is None else sigma

    # ========== 修改：支持0-255和0-1两种范围的阈值 ==========
    # 判断阈值范围并进行适当处理
    if low_thresh > 1:  # 大于1认为是0-255范围
        low_255 = int(np.clip(low_thresh, 0, 255))
    else:  # 小于等于1认为是0-1范围
        low_255 = int(np.clip(low_thresh * 255, 0, 255))
        
    if high_thresh > 1:  # 大于1认为是0-255范围
        high_255 = int(np.clip(high_thresh, 0, 255))
    else:  # 小于等于1认为是0-1范围
        high_255 = int(np.clip(high_thresh * 255, 0, 255))
    
    # 确保低阈值小于高阈值
    if low_255 >= high_255:
        raise ValueError(f"低阈值({low_255})不能大于等于高阈值({high_255})")
    if sigma <= 0:
        raise ValueError(f"高斯sigma需大于0，当前值：{sigma}")

    # ========== 输入图像预处理 ==========
    img_input = img.copy()
    if img_input.dtype == np.uint8:  # 若输入是0-255的uint8，转为0-1浮点型
        img_input = img_input / 255.0
    elif img_input.dtype not in [np.float32, np.float64]:  # 其他类型强制转浮点
        img_input = img_input.astype(np.float64)
    # 裁剪数值范围（防止超出0-1导致后续计算错误）
    img_input = np.clip(img_input, 0.0, 1.0)

    # ========== RGB转灰度 ==========
    if len(img_input.shape) == 3 and img_input.shape[2] == 3:
        # 先将float64转为uint8（0-255）→ 转灰度 → 再转回float64（0-1）
        img_uint8 = (img_input * 255).astype(np.uint8)
        gray_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)  # 此时输入是uint8，支持！
        gray_img = gray_uint8 / 255.0  # 转回0-1 float64
    elif len(img_input.shape) == 2:  # 灰度图直接使用
        gray_img = img_input.copy()
    else:
        raise ValueError(f"不支持的图像维度：{img_input.shape}（仅支持2D灰度/3D RGB彩色）")

    # ========== 高斯平滑 ==========
    kernel_size = 2 * int(np.ceil(2 * sigma)) + 1
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    smoothed_img = cv2.GaussianBlur(
        gray_img,
        ksize=(kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE
    )

    # ========== Canny 边缘检测 ==========
    # 直接使用已经转换好的255范围阈值
    canny_uint8 = cv2.Canny(
        (smoothed_img * 255).astype(np.uint8),
        threshold1=low_255,
        threshold2=high_255,
        L2gradient=True
    )
    edge_mask = canny_uint8 > 0

    # ========== 生成可视化边缘图 ==========
    edge_img = np.zeros_like(gray_img, dtype=np.float64)
    edge_img[edge_mask] = 1.0

    return edge_mask, edge_img