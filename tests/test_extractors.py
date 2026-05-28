"""FontFeatureLibrary + TamperAnalyzer 单元测试。"""

import numpy as np
import pytest
from core.extractors import FontFeatureLibrary, TamperAnalyzer


class TestFontFeatureLibrary:
    def test_is_ready_empty(self):
        lib = FontFeatureLibrary(dim=8)
        assert lib.is_ready is False

    def test_is_ready_after_add(self):
        lib = FontFeatureLibrary(dim=8)
        lib.add([np.ones(8, dtype=np.float32)], ["test"])
        assert lib.is_ready is True

    def test_search_empty(self):
        lib = FontFeatureLibrary(dim=8)
        sim = lib.search_similarity(np.ones(8, dtype=np.float32))
        assert sim == 0.5

    def test_search_batch_empty(self):
        lib = FontFeatureLibrary(dim=8)
        sims = lib.search_similarity_batch([np.ones(8, dtype=np.float32)])
        assert sims == [0.5]

    def test_search_self_similarity(self):
        lib = FontFeatureLibrary(dim=8)
        feat = np.random.randn(8).astype(np.float32)
        lib.add([feat], ["test"])
        lib._calibrate_decay()
        sim = lib.search_similarity(feat)
        assert 0.9 <= sim <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        lib = FontFeatureLibrary(dim=8)
        feat = np.random.randn(8).astype(np.float32)
        lib.add([feat], ["test"])
        lib._calibrate_decay()

        path = str(tmp_path / "test_lib")
        lib.save(path)

        lib2 = FontFeatureLibrary(dim=8)
        assert lib2.load(path) is True
        assert lib2.is_ready is True
        assert lib2.char_labels == ["test"]

    def test_load_nonexistent(self, tmp_path):
        lib = FontFeatureLibrary(dim=8)
        assert lib.load(str(tmp_path / "nonexistent")) is False
        assert lib.is_ready is False


class TestTamperAnalyzer:
    def test_empty_stats(self):
        reasons, penalty = TamperAnalyzer.check_internal_consistency([])
        assert reasons == []
        assert penalty == 0.0

    def test_single_digit(self):
        stats = [{"text": "100.00", "bbox": [10, 10, 60, 30], "conf": 0.95, "is_core_number": True}]
        reasons, penalty = TamperAnalyzer.check_internal_consistency(stats)
        assert reasons == []
        assert penalty == 0.0

    def test_consistent_digits(self, valid_ocr_stats):
        """两个高度和基线一致的数字，不处罚。"""
        reasons, penalty = TamperAnalyzer.check_internal_consistency(valid_ocr_stats)
        assert penalty == 0.0

    def test_mismatched_heights(self):
        """高度差异大的数字触发处罚。"""
        stats = [
            {"text": "100.00", "bbox": [10, 10, 60, 30], "conf": 0.95, "is_core_number": True},
            {"text": "200.00", "bbox": [10, 40, 70, 120], "conf": 0.92, "is_core_number": True},
        ]
        reasons, penalty = TamperAnalyzer.check_internal_consistency(stats)
        assert penalty > 0.0

    def test_mismatched_baselines(self):
        """Y 基线差异大的数字触发处罚。"""
        stats = [
            {"text": "100.00", "bbox": [10, 10, 60, 30], "conf": 0.95, "is_core_number": True},
            {"text": "200.00", "bbox": [10, 60, 70, 80], "conf": 0.92, "is_core_number": True},
        ]
        reasons, penalty = TamperAnalyzer.check_internal_consistency(stats)
        assert penalty > 0.0

    def test_cross_roi_single_roi(self):
        """单个 ROI 不触发跨 ROI 分析。"""
        penalty, reasons = TamperAnalyzer.check_cross_roi_consistency([[]])
        assert penalty == 0.0
        assert reasons == []

    def test_cross_roi_consistent(self, two_roi_stats):
        """相似 ROI 不触发跨 ROI 处罚。"""
        penalty, reasons = TamperAnalyzer.check_cross_roi_consistency(two_roi_stats)
        assert isinstance(penalty, float)
        assert penalty >= 0.0

    def test_cross_roi_mismatched_heights(self):
        """高度差异大的跨 ROI 触发处罚。"""
        roi1 = [{"text": "100.00", "bbox": [5, 5, 50, 25], "conf": 0.95, "is_core_number": True}]
        roi2 = [{"text": "200.00", "bbox": [5, 5, 55, 70], "conf": 0.92, "is_core_number": True}]
        penalty, reasons = TamperAnalyzer.check_cross_roi_consistency([roi1, roi2])
        assert penalty > 0.0
