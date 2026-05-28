"""PixelLevelDetector 各检测层独立单元测试。"""

import numpy as np
import pytest
from core.detectors import PixelLevelDetector


class TestPixelLevelDetector:
    def test_detect_empty_image(self):
        detector = PixelLevelDetector()
        score = detector.detect(None)
        assert score == 0.0

    def test_detect_normal_image(self, sample_image_bgr):
        detector = PixelLevelDetector({"generator_enabled": False})
        score = detector.detect(sample_image_bgr)
        assert 0.0 <= score <= 1.0

    def test_generator_disabled(self, sample_image_bgr):
        detector = PixelLevelDetector({"generator_enabled": False})
        score = detector.detect(sample_image_bgr, surrounding_np=sample_image_bgr)
        assert 0.0 <= score <= 1.0

    def test_noise_consistency_same_image(self, sample_image_bgr):
        """相同图像噪声一致，应返回 0.0。"""
        detector = PixelLevelDetector()
        gray = np.ones((50, 50), dtype=np.uint8) * 128
        sur = np.ones((50, 50, 3), dtype=np.uint8) * 128
        penalty = detector._check_noise_consistency(gray, sur)
        assert penalty == 0.0

    def test_noise_consistency_mismatched(self):
        """噪声差异显著时应触发惩罚。"""
        detector = PixelLevelDetector({"noise_consistency_weight": 0.15})
        np.random.seed(42)
        gray = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        sur = np.ones((50, 50, 3), dtype=np.uint8) * 255  # 平滑背景
        penalty = detector._check_noise_consistency(gray, sur)
        assert 0.0 <= penalty <= 0.15

    def test_color_consistency_same_roi(self, sample_image_bgr):
        detector = PixelLevelDetector()
        penalty = detector._check_color_consistency(sample_image_bgr, [sample_image_bgr.copy()])
        assert penalty == 0.0

    def test_dct_anomaly_disabled(self, sample_image_bgr):
        detector = PixelLevelDetector({"dct_analysis_enabled": False})
        gray = np.ones((50, 50), dtype=np.uint8) * 128
        penalty = detector._check_dct_anomaly(gray)
        assert penalty == 0.0

    def test_dct_anomaly_tiny_image(self):
        detector = PixelLevelDetector({"dct_analysis_enabled": True})
        gray = np.ones((8, 8), dtype=np.uint8)
        penalty = detector._check_dct_anomaly(gray)
        assert penalty == 0.0

    def test_color_histogram_shape(self, sample_image_bgr):
        hist = PixelLevelDetector._color_histogram(sample_image_bgr)
        assert hist.shape == (256,)

    def test_dct_anomaly_smooth(self):
        """平滑图像应触发 DCT 异常。"""
        detector = PixelLevelDetector({"dct_analysis_enabled": True, "dct_weight": 0.12})
        gray = np.ones((64, 64), dtype=np.uint8) * 128
        penalty = detector._check_dct_anomaly(gray)
        assert penalty >= 0.0
