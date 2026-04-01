import cv2
import numpy as np
from stitcher.common.logger import get_logger
from stitcher.config import (
    FEATURE_RATIO_THRESH,
    HIST_SHIFT_BIN_RATIO,
    RANSAC_REPROJ_THRESH,
    FEATURE_DETECTOR,
    FEATURE_NPOINTS,
    FEATURE_EDGE_THRESH,
    FEATURE_FIRST_LEVEL,
    FEATURE_N_OCTAVE_LAYERS,
    FEATURE_PATCH_SIZE,
    SIFT_EDGE_THRESH,
    AKAZE_THRESHOLD,
    FLANN_INDEX_PARAMS,
    FLANN_SEARCH_PARAMS,
    USE_FLANN
)

logger = get_logger(__name__)


def create_feature_detector(detector_type=None):
    detector_type = detector_type or FEATURE_DETECTOR
    
    if detector_type.upper() == 'SIFT':
        return cv2.SIFT_create(
            contrastThreshold=0.0,
            edgeThreshold=SIFT_EDGE_THRESH
        )
    elif detector_type.upper() == 'ORB':
        return cv2.ORB_create(
            nfeatures=FEATURE_NPOINTS,
            edgeThreshold=FEATURE_EDGE_THRESH,
            firstLevel=FEATURE_FIRST_LEVEL,
            nlevels=FEATURE_N_OCTAVE_LAYERS,
            patchSize=FEATURE_PATCH_SIZE
        )
    elif detector_type.upper() == 'AKAZE':
        return cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT,
            threshold=AKAZE_THRESHOLD,
            nOctaves=FEATURE_N_OCTAVE_LAYERS,
            nOctaveLayers=4
        )
    else:
        logger.warning(f"Unknown detector type: {detector_type}, falling back to ORB")
        return cv2.ORB_create(nfeatures=FEATURE_NPOINTS)


def create_matcher(descriptor_type='ORB'):
    if USE_FLANN:
        try:
            if descriptor_type in ['ORB', 'AKAZE']:
                return cv2.FlannBasedMatcher(FLANN_INDEX_PARAMS, FLANN_SEARCH_PARAMS)
            else:
                index_params = dict(algorithm=1, trees=5)
                return cv2.FlannBasedMatcher(index_params, FLANN_SEARCH_PARAMS)
        except Exception as e:
            logger.warning(f"Failed to create FLANN matcher: {e}, falling back to BFMatcher")
    
    if descriptor_type in ['ORB', 'AKAZE']:
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        return cv2.BFMatcher()


def _build_detector_try_order(detector_type):
    detectors_to_try = [detector_type] if detector_type else []
    
    if 'ORB' not in detectors_to_try:
        detectors_to_try.append('ORB')
    if 'SIFT' not in detectors_to_try:
        detectors_to_try.append('SIFT')
    if 'AKAZE' not in detectors_to_try:
        detectors_to_try.append('AKAZE')
    
    return detectors_to_try


def _detect_and_match(gray1, gray2, det_type):
    detector = create_feature_detector(det_type)
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return None, None, None, None, f"{det_type} 特征检测失败"
    
    matcher = create_matcher(det_type)
    matches = matcher.knnMatch(des1, des2, k=2)
    
    good = []
    for match_pair in matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < FEATURE_RATIO_THRESH * n.distance:
            good.append(m)
    
    if len(good) < 8:
        return None, None, None, None, f"{det_type} 匹配点不足"
    
    return kp1, kp2, good, det_type, None


def _deduplicate_matches(kp1, kp2, good_matches):
    pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    _, idx = np.unique(pts1, axis=0, return_index=True)
    pts1, pts2 = pts1[idx], pts2[idx]

    _, idx = np.unique(pts2, axis=0, return_index=True)
    pts1, pts2 = pts1[idx], pts2[idx]

    if len(pts1) < 8:
        raise RuntimeError("Too few matches after unique filter")
    
    return pts1, pts2


def _estimate_homography(pts1, pts2, img1_shape):
    pts1, pts2 = _histogram_filter(
        pts1, pts2,
        img1_shape,
        thr=HIST_SHIFT_BIN_RATIO
    )

    if len(pts1) < 8:
        raise RuntimeError("Too few matches after histogram filter")

    H0, mask = cv2.findHomography(
        pts2, pts1,
        cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESH
    )

    if H0 is None:
        raise RuntimeError("RANSAC failed")

    in1 = pts1[mask.ravel() == 1]
    in2 = pts2[mask.ravel() == 1]

    logger.info(f"Homography inliers: {len(in1)}")

    if len(in1) < 8:
        raise RuntimeError("Too few RANSAC inliers")

    H = _calc_homography_normalized(in2, in1)
    H = np.linalg.inv(H)
    return H


def _histogram_filter(pts1, pts2, shape, thr=0.1):
    h, w = shape[:2]

    dx = pts1[:, 0] - pts2[:, 0]
    dy = pts1[:, 1] - pts2[:, 1]

    xbins = np.arange(-w, w + 1e-6, w * thr)
    ybins = np.arange(-h, h + 1e-6, h * thr)

    hx, _ = np.histogram(dx, bins=xbins)
    hy, _ = np.histogram(dy, bins=ybins)

    ix = np.argmax(hx)
    iy = np.argmax(hy)

    x0 = max(ix - 2, 0)
    x1 = min(ix + 1, len(xbins) - 2)

    y0 = max(iy - 2, 0)
    y1 = min(iy + 1, len(ybins) - 2)

    mask = (
        (dx >= xbins[x0]) & (dx <= xbins[x1 + 1]) &
        (dy >= ybins[y0]) & (dy <= ybins[y1 + 1])
    )
    logger.info(f"Matches after hist filter: {len(pts1[mask])}")

    return pts1[mask], pts2[mask]


def _normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)

    T = np.array([
        [1 / std, 0, -mean[0] / std],
        [0, 1 / std, -mean[1] / std],
        [0, 0, 1]
    ])

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_n = (T @ pts_h.T).T

    return pts_n, T


def _calc_homography_normalized(pts_src, pts_dst):
    pts1_n, T1 = _normalize_points(pts_src)
    pts2_n, T2 = _normalize_points(pts_dst)

    A = []
    for (x, y), (u, v) in zip(pts1_n[:, :2], pts2_n[:, :2]):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T2) @ Hn @ T1
    return H / H[2, 2]


def registerTexture(img1, edge1, img2, edge2, detector_type=None):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    gray1 = (gray1 * edge1).astype(np.uint8)
    gray2 = (gray2 * edge2).astype(np.uint8)

    detector_type = (detector_type or FEATURE_DETECTOR).upper()
    detectors_to_try = _build_detector_try_order(detector_type)

    last_error = None
    success_result = None
    
    for det_type in detectors_to_try:
        try:
            logger.info(f"尝试使用 {det_type} 检测器")
            kp1, kp2, good, det_success, error_msg = _detect_and_match(gray1, gray2, det_type)
            
            if error_msg:
                last_error = error_msg
                continue
            
            logger.info(f"成功使用 {det_type} 检测器，找到 {len(good)} 个匹配点")
            pts1, pts2 = _deduplicate_matches(kp1, kp2, good)
            h_matrix = _estimate_homography(pts1, pts2, img1.shape)
            success_result = h_matrix
            break
            
        except Exception as e:
            last_error = f"{det_type} 处理失败: {str(e)}"
            logger.warning(last_error)
            continue
    
    if success_result is not None:
        return success_result
    else:
        raise RuntimeError(f"所有特征检测器都失败: {last_error}")
