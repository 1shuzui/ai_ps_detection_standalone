"""InferenceEngineAPI 单位测试：bbox 归一化、文本分析、异常处理。"""

import numpy as np
import pytest
from inference_api import InferenceEngineAPI


class TestBBoxNormalization:
    """bbox 格式归一化测试（无需加载模型）。"""

    @staticmethod
    def _make_engine():
        # 用虚拟实例测试静态/纯逻辑方法 — 不加载实际模型
        return InferenceEngineAPI.__new__(InferenceEngineAPI)

    def test_xyxy_format(self):
        engine = self._make_engine()
        result = engine._clip_bbox_xyxy([10, 20, 100, 200], 800, 600)
        assert result == (10, 20, 100, 200)

    def test_clip_out_of_bounds(self):
        engine = self._make_engine()
        x1, y1, x2, y2 = engine._clip_bbox_xyxy([-5, -10, 900, 700], 800, 600)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 800
        assert y2 == 600

    def test_bbox_too_small(self):
        engine = self._make_engine()
        x1, y1, x2, y2 = engine._clip_bbox_xyxy([100, 100, 100, 100], 800, 600)
        assert x2 > x1
        assert y2 > y1

    def test_normalize_xywh(self):
        engine = self._make_engine()
        x1, y1, x2, y2 = engine._normalize_roi_bbox([10, 20, 100, 50], 800, 600, "xywh")
        assert x1 == 10
        assert y1 == 20
        assert x2 == 110  # 10 + 100
        assert y2 == 70   # 20 + 50

    def test_normalize_auto_looks_like_xyxy(self):
        engine = self._make_engine()
        x1, y1, x2, y2 = engine._normalize_roi_bbox([10, 20, 200, 100], 800, 600, "auto")
        # 200 > 10 and 100 > 20, within bounds → treated as xyxy
        assert x1 == 10
        assert y1 == 20
        assert x2 == 200
        assert y2 == 100

    def test_normalize_invalid_bbox_length(self):
        engine = self._make_engine()
        with pytest.raises(ValueError):
            engine._normalize_roi_bbox([10, 20, 30], 800, 600, "auto")


class TestTextProfile:
    @staticmethod
    def _make_engine():
        return InferenceEngineAPI.__new__(InferenceEngineAPI)

    def test_core_amount_text(self):
        engine = self._make_engine()
        profile = engine._profile_numeric_text("¥100.50", 15)
        assert profile["digit_count"] == 5
        assert profile["is_core_candidate"] == 1.0
        assert profile["should_use_font_signal"] == 1.0

    def test_non_numeric_text(self):
        engine = self._make_engine()
        profile = engine._profile_numeric_text("hello world", 15)
        assert profile["digit_count"] == 0
        assert profile["is_core_candidate"] == 0.0
        assert profile["should_use_font_signal"] == 0.0

    def test_order_number_text(self):
        engine = self._make_engine()
        profile = engine._profile_numeric_text("订单号:20240501123456", 15)
        assert profile["digit_count"] == 14
        # 长数字带订单关键词也算核心候选
        assert profile["is_core_candidate"] == 1.0
