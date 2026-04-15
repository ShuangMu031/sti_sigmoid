import cv2
import numpy as np
import time


try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
from utils import get_logger

logger = get_logger(__name__)

# 全局变量初始化 - 确保窗口管理变量在模块加载时就被定义
# 用于给每次显示的显著图创建唯一窗口名并定位，避免被覆盖
_SAL_WIN_COUNTER = 0
# 用于存储所有创建的窗口名称，便于后续管理和清理
_SAL_WINDOWS = []


# ===================== 核心参数配置（复刻getParam） =====================
def getParam():
    """
    完整复刻MATLAB的getParam()：MBS显著性检测参数配置
    对应论文 "Minimum Barrier Salient Object Detection at 80 FPS"
    """
    param = {
        # 基础配置
        'verbose': False,  # 过程信息输出
        'MAX_DIM': 400,  # 图像缩放最大维度（平衡速度/精度）
        'use_backgroundness': True,  # 是否融合背景图增强显著性
        'use_geodesic': False,  # 是否使用Geodesic距离（False=MB模式，True=Geodesic模式）
        'remove_border': True,  # 是否移除图像边框（避免边框干扰）
        'colorSpace': 'LAB',  # 颜色空间：LAB（MBS默认）
        'sigma_smooth': 1.0,  # 后处理平滑系数
        'contrast_enhance': True,  # 是否增强对比度

        # MBS核心参数
        'theta': 0.1,  # 障碍阈值（Geodesic模式）
        'num_iter': 2,  # 距离计算迭代次数
        'erosion_size': 3,  # 形态学腐蚀核大小
        'dilation_size': 3  # 形态学膨胀核大小
    }
    return param


# ===================== 背景图计算（复刻BG.m） =====================
def BG(I):
    """
    复刻MATLAB的BG.m：计算背景图（Backgroundness）
    输入：I - 0-1的RGB图像（np.float64）
    输出：bg - 0-1的背景图（np.float64）
    """
    # 转换为灰度图
    I_gray = cv2.cvtColor((I * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
    h, w = I_gray.shape

    # 步骤1：计算四个边缘区域的均值（上下左右各10%）
    border_h = int(0.1 * h)
    border_w = int(0.1 * w)
    # 上边缘
    top = I_gray[:border_h, :].mean()
    # 下边缘
    bottom = I_gray[-border_h:, :].mean()
    # 左边缘
    left = I_gray[:, :border_w].mean()
    # 右边缘
    right = I_gray[:, -border_w:].mean()

    # 步骤2：生成背景图（距离边缘越近，背景值越高）
    bg = np.zeros_like(I_gray)
    for y in range(h):
        for x in range(w):
            # 计算到四个边缘的归一化距离
            d_top = y / h
            d_bottom = (h - y - 1) / h
            d_left = x / w
            d_right = (w - x - 1) / w
            # 背景值 = 边缘均值 * 距离权重
            bg[y, x] = (top * (1 - d_top) + bottom * (1 - d_bottom) + left * (1 - d_left) + right * (1 - d_right)) / 4.0

    # 步骤3：归一化到0-1
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
    return bg


# ===================== MBS核心距离计算（复刻fastMBS） =====================
if _NUMBA_AVAILABLE:
    @njit
    def fastMBS(I):
        h, w = I.shape
        dist = np.ones((h, w), dtype=np.float64) * 1e9

        dist[0, :] = 0
        dist[-1, :] = 0
        dist[:, 0] = 0
        dist[:, -1] = 0

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                d1 = dist[y - 1, x] + abs(I[y, x] - I[y - 1, x])
                d2 = dist[y, x - 1] + abs(I[y, x] - I[y, x - 1])
                d3 = dist[y - 1, x - 1] + abs(I[y, x] - I[y - 1, x - 1])
                m = d1
                if d2 < m:
                    m = d2
                if d3 < m:
                    m = d3
                dist[y, x] = m

        for y in range(h - 2, 0, -1):
            for x in range(w - 2, 0, -1):
                d1 = dist[y + 1, x] + abs(I[y, x] - I[y + 1, x])
                d2 = dist[y, x + 1] + abs(I[y, x] - I[y, x + 1])
                d3 = dist[y + 1, x + 1] + abs(I[y, x] - I[y + 1, x + 1])
                m = dist[y, x]
                if d1 < m:
                    m = d1
                if d2 < m:
                    m = d2
                if d3 < m:
                    m = d3
                dist[y, x] = m

        dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
        dist = 1 - dist
        return dist
else:
    def fastMBS(I):
        """
        复刻C++ mex的fastMBS：最小障碍距离（Minimum Barrier）计算
        输入：I - 0-1的LAB单通道图像（np.float64）
        输出：dist - 归一化的距离图（np.float64）
        """
        h, w = I.shape
        dist = np.ones((h, w), dtype=np.float64) * 1e9  # 初始化距离为极大值

        # 步骤1：设置边界点为起始点（距离=0）
        dist[0, :] = 0
        dist[-1, :] = 0
        dist[:, 0] = 0
        dist[:, -1] = 0

        # 步骤2：Raster扫描（从上到下，从左到右）
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                # 邻域距离（上/左/左上）
                d1 = dist[y - 1, x] + abs(I[y, x] - I[y - 1, x])
                d2 = dist[y, x - 1] + abs(I[y, x] - I[y, x - 1])
                d3 = dist[y - 1, x - 1] + abs(I[y, x] - I[y - 1, x - 1])
                dist[y, x] = min(d1, d2, d3)

        # 步骤3：逆Raster扫描（从下到上，从右到左）
        for y in range(h - 2, 0, -1):
            for x in range(w - 2, 0, -1):
                # 邻域距离（下/右/右下）
                d1 = dist[y + 1, x] + abs(I[y, x] - I[y + 1, x])
                d2 = dist[y, x + 1] + abs(I[y, x] - I[y, x + 1])
                d3 = dist[y + 1, x + 1] + abs(I[y, x] - I[y + 1, x + 1])
                dist[y, x] = min(dist[y, x], d1, d2, d3)

        # 步骤4：归一化距离图（0-1）
        dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
        # 反转：距离越大（越远离边界），显著性越高
        dist = 1 - dist
        return dist

# ===================== Geodesic距离计算（复刻fastGeodesic） =====================
if _NUMBA_AVAILABLE:
    @njit
    def fastGeodesic(I, theta=0.1):
        h, w = I.shape
        dist = np.ones((h, w), dtype=np.float64) * 1e9
        dist[0, :] = 0
        dist[-1, :] = 0
        dist[:, 0] = 0
        dist[:, -1] = 0

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                d1 = dist[y - 1, x] + (abs(I[y, x] - I[y - 1, x]) > theta)
                d2 = dist[y, x - 1] + (abs(I[y, x] - I[y, x - 1]) > theta)
                d3 = dist[y - 1, x - 1] + (abs(I[y, x] - I[y - 1, x - 1]) > theta)
                m = d1
                if d2 < m:
                    m = d2
                if d3 < m:
                    m = d3
                dist[y, x] = m

        for y in range(h - 2, 0, -1):
            for x in range(w - 2, 0, -1):
                d1 = dist[y + 1, x] + (abs(I[y, x] - I[y + 1, x]) > theta)
                d2 = dist[y, x + 1] + (abs(I[y, x] - I[y, x + 1]) > theta)
                d3 = dist[y + 1, x + 1] + (abs(I[y, x] - I[y + 1, x + 1]) > theta)
                m = dist[y, x]
                if d1 < m:
                    m = d1
                if d2 < m:
                    m = d2
                if d3 < m:
                    m = d3
                dist[y, x] = m

        dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
        dist = 1 - dist
        return dist
else:
    def fastGeodesic(I, theta=0.1):
        h, w = I.shape
        dist = np.ones((h, w), dtype=np.float64) * 1e9
        # 边界点初始化
        dist[0, :] = 0
        dist[-1, :] = 0
        dist[:, 0] = 0
        dist[:, -1] = 0

        # Raster扫描
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                d1 = dist[y - 1, x] + (abs(I[y, x] - I[y - 1, x]) > theta)
                d2 = dist[y, x - 1] + (abs(I[y, x] - I[y, x - 1]) > theta)
                d3 = dist[y - 1, x - 1] + (abs(I[y, x] - I[y - 1, x - 1]) > theta)
                dist[y, x] = min(d1, d2, d3)

        # 逆Raster扫描
        for y in range(h - 2, 0, -1):
            for x in range(w - 2, 0, -1):
                d1 = dist[y + 1, x] + (abs(I[y, x] - I[y + 1, x]) > theta)
                d2 = dist[y, x + 1] + (abs(I[y, x] - I[y, x + 1]) > theta)
                d3 = dist[y + 1, x + 1] + (abs(I[y, x] - I[y + 1, x + 1]) > theta)
                dist[y, x] = min(dist[y, x], d1, d2, d3)

        # 归一化+反转
        dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
        dist = 1 - dist
        return dist


# ===================== 核心处理函数（复刻doMBS.m） =====================
def doMBS(I, param):
    """
    完整复刻MATLAB的doMBS.m：MBS显著性检测主流程
    输入：
        I     - 0-1的RGB图像（np.float64）
        param - 参数配置（getParam()返回）
    输出：
        pMap  - 0-1的显著性图（np.float64）
    """
    if param['verbose']:
        logger.info("[doMBS] 开始MBS显著性检测...")
        start_total = time.time()

    h, w = I.shape[:2]
    # 步骤1：图像缩放（统一到MAX_DIM）
    scale = param['MAX_DIM'] / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    I_resize = cv2.resize(I, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 步骤2：颜色空间转换（LAB）
    I_uint8 = (I_resize * 255).astype(np.uint8)
    I_lab = cv2.cvtColor(I_uint8, cv2.COLOR_BGR2LAB) / 255.0
    # 取L通道（亮度通道，MBS核心特征）
    I_l = I_lab[:, :, 0]

    # 步骤3：距离计算（MB/Geodesic）
    if param['verbose']:
        start_dist = time.time()
    if param['use_geodesic']:
        dist = fastGeodesic(I_l, param['theta'])
    else:
        dist = fastMBS(I_l)
        if param['verbose']:
            logger.info(f"[doMBS] 距离计算耗时：{time.time() - start_dist:.4f}s")

    # 步骤4：背景图融合（可选）
    if param['use_backgroundness']:
        if param['verbose']:
            start_bg = time.time()
        bg = BG(I_resize)
        # 融合：显著性 = 距离图 * (1 - 背景图)
        dist = dist * (1 - bg)
        if param['verbose']:
            logger.info(f"[doMBS] 背景图融合耗时：{time.time() - start_bg:.4f}s")

    # 步骤5：后处理（形态学平滑+对比度增强）
    if param['verbose']:
        start_post = time.time()
    # 缩放回原始尺寸
    dist = cv2.resize(dist, (w, h), interpolation=cv2.INTER_LINEAR)

    # 移除边框（可选）
    if param['remove_border']:
        border = int(0.05 * min(h, w))
        dist[:border, :] = 0
        dist[-border:, :] = 0
        dist[:, :border] = 0
        dist[:, -border:] = 0

    # 高斯平滑
    dist = cv2.GaussianBlur(dist, (0, 0), sigmaX=param['sigma_smooth'])

    # 形态学操作（开运算+闭运算）
    kernel_erode = np.ones((param['erosion_size'], param['erosion_size']), np.uint8)
    kernel_dilate = np.ones((param['dilation_size'], param['dilation_size']), np.uint8)
    dist_uint8 = (dist * 255).astype(np.uint8)
    dist_uint8 = cv2.morphologyEx(dist_uint8, cv2.MORPH_OPEN, kernel_erode)
    dist_uint8 = cv2.morphologyEx(dist_uint8, cv2.MORPH_CLOSE, kernel_dilate)
    dist = dist_uint8 / 255.0

    # 对比度增强（可选）
    if param['contrast_enhance']:
        dist = cv2.equalizeHist((dist * 255).astype(np.uint8)) / 255.0

    # 最终归一化
    pMap = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)

    # 移除中间结果显示环节，直接返回处理后的显著性图

    if param['verbose']:
        logger.info(f"[doMBS] 后处理耗时：{time.time() - start_post:.4f}s")
        logger.info(f"[doMBS] 总耗时：{time.time() - start_total:.4f}s")

    return pMap


# ===================== 对外接口（复刻mbs_saliency.m） =====================
def mbs_saliency(img):
    """
    完整复刻MATLAB的mbs_saliency.m
    输入：img - 0-1的RGB图像（np.float64）
    输出：pMap - 0-1的显著性图（np.float64）
    """
    # 步骤1：获取参数配置
    paramMBplus = getParam()

    # 步骤2：开启过程输出
    paramMBplus['verbose'] = True

    # 步骤3：复制图像（避免修改原始数据）
    I = img.copy()

    # 步骤4：调用核心算法
    pMap = doMBS(I, paramMBplus)

    # 移除图像显示功能，直接返回显著性图
    return pMap


def wait_saliency_windows():
    """阻塞直到用户按键，然后关闭所有显著图窗口。
    调用场景：当你希望在程序末尾让显著图保持可见直到手动关闭时，调用此函数。
    """
    if len(_SAL_WINDOWS) == 0:
        return
    logger.info(f"等待手动关闭 {len(_SAL_WINDOWS)} 个显著图窗口...")
    # 阻塞直到按键
    cv2.waitKey(0)
    # 依次销毁窗口
    for wn in list(_SAL_WINDOWS):
        try:
            cv2.destroyWindow(wn)
        except Exception:
            pass
    _SAL_WINDOWS.clear()