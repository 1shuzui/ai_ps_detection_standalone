"""OCR 解析与金额候选框构建单元测试。"""

import numpy as np
import pytest
from core.ocr_utils import (
    OCRToken,
    AmountCandidate,
    AMOUNT_PATTERN,
    bbox_iou,
    normalize_text,
    score_amount_text,
    looks_like_clock_time,
    is_viable_amount_candidate,
    tokenize_ocr_results,
)


class TestNormalizeText:
    def test_basic_clean(self):
        assert normalize_text("¥100.50") == "¥100.50"

    def test_replace_chinese_punctuation(self):
        assert normalize_text("金额：100.50") == "金额:100.50"

    def test_replace_yuan_variant(self):
        assert normalize_text("100尤") == "100元"


class TestBBoxIOU:
    def test_identical(self):
        assert bbox_iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0

    def test_no_overlap(self):
        assert bbox_iou((0, 0, 50, 50), (100, 100, 150, 150)) == 0.0

    def test_partial_overlap(self):
        iou = bbox_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.0 < iou < 1.0


class TestAmountPattern:
    def test_match_yen_amount(self):
        assert AMOUNT_PATTERN.search("¥100.50")

    def test_match_yuan_amount(self):
        assert AMOUNT_PATTERN.search("100.00元")

    def test_no_match_non_amount(self):
        assert not AMOUNT_PATTERN.search("hello world")

    def test_match_with_comma(self):
        assert AMOUNT_PATTERN.search("1,234.56")


class TestScoreAmountText:
    def test_money_regex(self):
        score, flags = score_amount_text("¥100.50", (10, 10, 80, 30), (100, 200, 3))
        assert score > 0
        assert "money_regex" in flags

    def test_target_keyword(self):
        score, flags = score_amount_text("金额:100.00", (10, 10, 100, 30), (100, 200, 3))
        assert score > 0
        assert "target_keyword" in flags

    def test_no_digits(self):
        score, flags = score_amount_text("hello", (10, 10, 50, 20), (100, 200, 3))
        assert score == 0.0
        assert flags == []


class TestLooksLikeClockTime:
    def test_valid_time(self):
        assert looks_like_clock_time("12:30") is True

    def test_valid_time_with_seconds(self):
        assert looks_like_clock_time("23:59:59") is True

    def test_invalid_hour(self):
        assert looks_like_clock_time("25:00") is False

    def test_not_a_time(self):
        assert looks_like_clock_time("100.50") is False


class TestIsViableAmountCandidate:
    def test_viable_with_money_regex(self):
        assert is_viable_amount_candidate("100.50", ["money_regex"]) is True

    def test_not_viable_no_money_evidence(self):
        assert is_viable_amount_candidate("123456", ["fallback_digits"]) is False

    def test_not_viable_masked_account(self):
        assert is_viable_amount_candidate("6222****1234", ["masked_account_penalty"]) is False


class TestTokenizeOCR:
    def test_basic_tokenization(self):
        ocr_results = [
            ([[0, 0], [50, 0], [50, 20], [0, 20]], "100.00", 0.95),
            ([[60, 0], [80, 0], [80, 20], [60, 20]], "元", 0.90),
        ]
        tokens = tokenize_ocr_results(ocr_results)
        assert len(tokens) == 2
        assert tokens[0].clean_text == "100.00"
        assert tokens[0].conf == 0.95

    def test_empty_text_skipped(self):
        ocr_results = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "   ", 0.50),
        ]
        tokens = tokenize_ocr_results(ocr_results)
        assert len(tokens) == 0
