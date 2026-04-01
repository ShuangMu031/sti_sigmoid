import numpy as np
import pytest
from stitcher.config import (
    FEATURE_DETECTOR,
    GC_SALIENCY_WEIGHT,
    GC_OBJECT_WEIGHT,
    GC_EDGE_PENALTY,
    GC_ROI_MARGIN,
    GC_OBJECT_THRESH,
    AKAZE_THRESHOLD,
    POISSON_USE_FACTORIZED,
    POISSON_NEIGHBOR_MODE
)
from stitcher.algorithms import (
    create_feature_detector,
    create_matcher,
    graph_cut_seam,
    gradient_blend_local
)


def test_config_values_exist():
    assert FEATURE_DETECTOR is not None
    assert GC_SALIENCY_WEIGHT == 3.0
    assert GC_OBJECT_WEIGHT == 4.0
    assert GC_EDGE_PENALTY == 5.0
    assert GC_ROI_MARGIN == 10
    assert GC_OBJECT_THRESH == 0.3
    assert AKAZE_THRESHOLD == 0.001
    assert POISSON_USE_FACTORIZED is True
    assert POISSON_NEIGHBOR_MODE == "target_neighbor"


def test_create_feature_detector():
    detector = create_feature_detector('ORB')
    assert detector is not None
    
    detector = create_feature_detector('SIFT')
    assert detector is not None
    
    detector = create_feature_detector('AKAZE')
    assert detector is not None


def test_create_matcher():
    matcher = create_matcher('ORB')
    assert matcher is not None
    
    matcher = create_matcher('SIFT')
    assert matcher is not None


def test_graphcut_returns_bool_mask():
    h, w = 100, 100
    img1 = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    sal1 = np.random.rand(h, w).astype(np.float32)
    sal2 = np.random.rand(h, w).astype(np.float32)
    edge1 = np.random.rand(h, w).astype(np.float32)
    edge2 = np.random.rand(h, w).astype(np.float32)
    valid1 = np.ones((h, w), dtype=bool)
    valid2 = np.ones((h, w), dtype=bool)
    overlap = np.ones((h, w), dtype=bool)
    
    select_img1_mask = graph_cut_seam(
        img1, img2, sal1, sal2, edge1, edge2, valid1, valid2, overlap
    )
    
    assert select_img1_mask.dtype == bool
    assert select_img1_mask.shape == (h, w)


def test_poisson_blend_handles_empty_mask():
    h, w = 50, 50
    source = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    target = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    
    result = gradient_blend_local(source, target, mask)
    
    assert result.shape == (h, w, 3)
    assert np.all(result == target)


if __name__ == '__main__':
    test_config_values_exist()
    print("✓ test_config_values_exist passed")
    
    test_create_feature_detector()
    print("✓ test_create_feature_detector passed")
    
    test_create_matcher()
    print("✓ test_create_matcher passed")
    
    print("\n所有测试通过！")
