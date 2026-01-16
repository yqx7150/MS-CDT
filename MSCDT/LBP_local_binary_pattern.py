import cv2
from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def extract_LBP_features(image_path, radius=3, n_points=8):
    img = image_path
    if img is None:
        print("无法读取图像，请检查图像路径。")
        return None

    img = img.astype(np.float32)
    gray = img_max_min_normalization(img)
    gray = gray.astype(np.uint8)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, n_points, radius, method='default')
    lbp_processed = np.where(lbp > 120, 1, 0)

    Mask_LBP = gray * lbp_processed

    return Mask_LBP

def img_max_min_normalization(src, min=0, max=255):
    height = src.shape[0]
    width = src.shape[1]
    if len(src.shape) > 2:
        channel = src.shape[2]
    else:
        channel = 1

    src_min = np.min(src)
    src_max = np.max(src)

    if channel == 1:
        dst = np.zeros([height, width], dtype=np.float32)
        for h in range(height):
            for w in range(width):
                dst[h, w] = float(src[h, w] - src_min) / float(src_max - src_min + 1e-6) * (max - min) + min
    else:
        dst = np.zeros([height, width, channel], dtype=np.float32)
        for c in range(channel):
            for h in range(height):
                for w in range(width):
                    dst[h, w, c] = float(src[h, w, c] - src_min) / float(src_max - src_min + 1e-6) * (max - min) + min

    return dst

