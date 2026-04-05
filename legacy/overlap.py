import numpy as np
import cv2

def compute_overlap_masks(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    A = gray1 > 0
    B = gray2 > 0
    C = A & B

    return A, B, C
