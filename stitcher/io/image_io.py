import os
import numpy as np
import cv2


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def cv_imwrite(file_path, img):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    ext = os.path.splitext(file_path)[1] or '.png'
    success, encoded = cv2.imencode(ext, img)
    if not success:
        return False
    encoded.tofile(file_path)
    return True


class ImageIOHandler:
    SUPPORTED_FORMATS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
        '.webp', '.jp2', '.pbm', '.pgm', '.ppm'
    }
    
    def __init__(self, default_output_dir='output'):
        self.default_output_dir = default_output_dir
        self._ensure_directory(default_output_dir)
    
    def load_image(self, image_path):
        if not os.path.exists(image_path):
            return None
        return cv_imread(image_path)
    
    def load_images(self, image_paths):
        return [self.load_image(p) for p in image_paths if self.load_image(p) is not None]
    
    def save_image(self, image, output_path, quality=95, create_dir=True):
        if create_dir:
            self._ensure_directory(os.path.dirname(output_path))
        return cv_imwrite(output_path, image)
    
    def _ensure_directory(self, directory):
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
