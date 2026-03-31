import os
import numpy as np


def ensure_directory(path):
    if path and not os.path.exists(path):
        os.makedirs(path)
    return path


def get_stitched_size(img1_w, img2_w):
    h = max(img1_w.shape[0], img2_w.shape[0])
    w = max(img1_w.shape[1], img2_w.shape[1])
    return h, w
