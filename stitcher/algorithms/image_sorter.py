import cv2
import numpy as np
from stitcher.common.logger import get_logger
from stitcher.algorithms.feature_registration import create_feature_detector, create_matcher
from stitcher.config import FEATURE_RATIO_THRESH

logger = get_logger(__name__)


def compute_pair_overlap(img1, img2, detector_type='ORB'):
    try:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        detector = create_feature_detector(detector_type)
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return 0.0
        
        matcher = create_matcher(detector_type)
        matches = matcher.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < FEATURE_RATIO_THRESH * n.distance:
                good_matches.append(m)
        
        total_keypoints = min(len(kp1), len(kp2))
        if total_keypoints == 0:
            return 0.0
        
        overlap_score = len(good_matches) / total_keypoints
        return overlap_score
    
    except Exception as e:
        logger.warning(f"Failed to compute overlap: {e}")
        return 0.0


def build_overlap_matrix(images, detector_type='ORB'):
    n = len(images)
    overlap_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            score = compute_pair_overlap(images[i], images[j], detector_type)
            overlap_matrix[i][j] = score
            overlap_matrix[j][i] = score
    
    return overlap_matrix


def find_optimal_order(overlap_matrix):
    n = overlap_matrix.shape[0]
    if n <= 1:
        return list(range(n))
    
    visited = set()
    order = []
    
    start_idx = 0
    max_sum = 0
    for i in range(n):
        current_sum = np.sum(overlap_matrix[i])
        if current_sum > max_sum:
            max_sum = current_sum
            start_idx = i
    
    order.append(start_idx)
    visited.add(start_idx)
    
    while len(order) < n:
        last_idx = order[-1]
        best_next = -1
        best_score = -1
        
        for i in range(n):
            if i not in visited and overlap_matrix[last_idx][i] > best_score:
                best_score = overlap_matrix[last_idx][i]
                best_next = i
        
        if best_next == -1:
            for i in range(n):
                if i not in visited:
                    best_next = i
                    break
        
        order.append(best_next)
        visited.add(best_next)
    
    return order


def sort_images_by_overlap(images, detector_type='ORB'):
    if len(images) <= 2:
        return list(range(len(images)))
    
    logger.info("Building overlap matrix...")
    overlap_matrix = build_overlap_matrix(images, detector_type)
    
    logger.info("Finding optimal stitching order...")
    optimal_order = find_optimal_order(overlap_matrix)
    
    logger.info(f"Optimal order: {optimal_order}")
    return optimal_order
