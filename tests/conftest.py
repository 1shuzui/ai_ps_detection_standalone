"""
共享 fixture: 样本图片生成、mock 模型。
"""

import numpy as np
import pytest


@pytest.fixture
def sample_image_rgb():
    """200x100 白色背景 RGB 图像。"""
    return np.ones((100, 200, 3), dtype=np.uint8) * 255


@pytest.fixture
def sample_image_bgr():
    """200x100 白色背景 BGR 图像。"""
    return np.ones((100, 200, 3), dtype=np.uint8) * 255


@pytest.fixture
def text_roi_bgr():
    """模拟包含黑色文字的 ROI 图像（BGR）。"""
    img = np.ones((32, 128, 3), dtype=np.uint8) * 255
    img[8:24, 10:118] = 0  # 黑色数字区域
    return img


@pytest.fixture
def empty_image():
    """空图像。"""
    return np.array([], dtype=np.uint8).reshape(0, 0, 3)


@pytest.fixture
def valid_ocr_stats():
    """模拟 OCR 提取的数字统计信息列表（同基线、相似高度）。"""
    return [
        {"text": "100.50", "bbox": [10, 10, 60, 30], "conf": 0.95, "is_core_number": True},
        {"text": "元", "bbox": [65, 10, 85, 30], "conf": 0.90, "is_core_number": False},
        {"text": "200.00", "bbox": [80, 12, 140, 32], "conf": 0.92, "is_core_number": True},
    ]


@pytest.fixture
def two_roi_stats():
    """两个独立 ROI 的 OCR 统计信息。"""
    roi1 = [
        {"text": "150.00", "bbox": [5, 5, 50, 25], "conf": 0.95, "is_core_number": True},
        {"text": "元", "bbox": [55, 5, 75, 25], "conf": 0.90, "is_core_number": False},
    ]
    roi2 = [
        {"text": "250.00", "bbox": [5, 5, 55, 28], "conf": 0.93, "is_core_number": True},
        {"text": "元", "bbox": [60, 5, 80, 28], "conf": 0.88, "is_core_number": False},
    ]
    return [roi1, roi2]
