"""
OCR 工具模块 — 从 OCR 原始结果中提取结构化文本、构建金额候选区域、规则化凭证检测。
从 batch_eval_amounts.py 抽取而来，供 main.py 和批量评估脚本共用。
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.detectors import OriginalityChecker

# ---- 正则常量 ----
AMOUNT_PATTERN = re.compile(r"(?<!\d)[+\-]?(?:[¥￥])?\d[\d,]{0,12}[.:]\d{2}(?:元)?")
DATE_PATTERN = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
TIME_PATTERN = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")
ORDER_PATTERN = re.compile(r"\d{10,}")
MASKED_ACCOUNT_PATTERN = re.compile(r"\d{3,}\*+\d{2,}")

# ---- 关键词常量 ----
TARGET_AMOUNT_KEYWORDS = ("金额", "小写", "转账金额", "交易金额", "收款金额", "付款金额", "支出", "收入", "到账", "转出", "转入")
GENERIC_CURRENCY_KEYWORDS = ("人民币", "¥", "￥", "元")
NON_TARGET_AMOUNT_KEYWORDS = ("红包", "手续费", "余额", "本次余额", "剩余", "免费提现", "待领取", "积分", "福利", "奖励", "账户余")
ORDER_KEYWORDS = ("单号", "订单", "流水", "参考号", "凭证号", "转账单号", "汇款流水号")

# ---- 凭证规则常量 ----
CERTIFICATE_HEADER_KEYWORDS = ("电子凭证", "转账电子凭证", "凭证")
CERTIFICATE_RULE_REASON = "电子凭证金额行结构异常"
CERTIFICATE_RULE_FLAGS = "certificate_header|certificate_amount_row|ocr_structure_anomaly"
CERTIFICATE_ROW_TEXT_OPTIONS = {
    "text_threshold": 0.2,
    "low_text": 0.2,
    "mag_ratio": 1.5,
}


@dataclass
class OCRToken:
    text: str
    clean_text: str
    bbox: Tuple[int, int, int, int]
    conf: float
    width: int
    height: int
    center_y: float


@dataclass
class AmountCandidate:
    source: str
    text: str
    clean_text: str
    bbox: Tuple[int, int, int, int]
    ocr_confidence: float
    amount_score: float
    match_flags: str


def normalize_text(text: str) -> str:
    return (
        text.replace(" ", "")
        .replace("，", ",")
        .replace("。", ".")
        .replace("：", ":")
        .replace("￥", "¥")
        .replace("尤", "元")
        .replace("丫", "¥")
    )


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter_area / max(1, area_a + area_b - inter_area)


def score_amount_text(text: str, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]) -> Tuple[float, List[str]]:
    clean_text = normalize_text(text)
    digit_count = len(re.findall(r"\d", clean_text))
    total_len = len(clean_text)
    height = max(1, bbox[3] - bbox[1])
    width = max(1, bbox[2] - bbox[0])
    image_h, image_w = image_shape[:2]

    if digit_count == 0 or total_len == 0:
        return 0.0, []

    flags: List[str] = []
    score = 0.0

    money_match = AMOUNT_PATTERN.search(clean_text)
    if money_match:
        score += 1.2
        flags.append("money_regex")

    if any(keyword in clean_text for keyword in TARGET_AMOUNT_KEYWORDS):
        score += 0.65
        flags.append("target_keyword")

    if any(keyword in clean_text for keyword in GENERIC_CURRENCY_KEYWORDS):
        score += 0.2
        flags.append("currency_hint")

    if digit_count >= 4 and total_len <= 16:
        score += 0.2
        flags.append("compact_digits")

    if height >= image_h * 0.025 or width >= image_w * 0.18:
        score += 0.15
        flags.append("prominent")

    if clean_text.startswith(("+", "-", "¥")):
        score += 0.15
        flags.append("signed_or_currency")

    if any(keyword in clean_text for keyword in NON_TARGET_AMOUNT_KEYWORDS):
        score -= 1.1
        flags.append("non_target_penalty")

    if bbox[1] <= image_h * 0.12 and not any(keyword in clean_text for keyword in TARGET_AMOUNT_KEYWORDS):
        score -= 0.35
        flags.append("top_ui_penalty")

    if bbox[1] <= image_h * 0.08 and TIME_PATTERN.fullmatch(clean_text):
        score -= 0.6
        flags.append("status_bar_time_penalty")

    if MASKED_ACCOUNT_PATTERN.search(clean_text):
        score -= 1.2
        flags.append("masked_account_penalty")

    if clean_text.count(":") >= 2:
        score -= 1.0
        flags.append("double_colon_penalty")
    elif ":" in clean_text and "." not in clean_text and not any(keyword in clean_text for keyword in TARGET_AMOUNT_KEYWORDS + GENERIC_CURRENCY_KEYWORDS) and not clean_text.startswith(("+", "-", "¥")):
        score -= 0.7
        flags.append("colon_time_penalty")

    if DATE_PATTERN.search(clean_text):
        score -= 0.9
        flags.append("date_penalty")

    if TIME_PATTERN.search(clean_text) and not money_match:
        score -= 0.5
        flags.append("time_penalty")

    if any(keyword in clean_text for keyword in ORDER_KEYWORDS):
        score -= 0.7
        flags.append("order_keyword_penalty")

    if ORDER_PATTERN.fullmatch(clean_text) and not money_match:
        score -= 0.7
        flags.append("long_digits_penalty")

    if re.search(r"[A-Za-z]", clean_text) and not any(keyword in clean_text for keyword in TARGET_AMOUNT_KEYWORDS + GENERIC_CURRENCY_KEYWORDS) and not clean_text.startswith(("+", "-", "¥")):
        score -= 0.8
        flags.append("latin_noise_penalty")

    return max(0.0, round(score, 4)), flags


def looks_like_clock_time(clean_text: str) -> bool:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", clean_text)
    if not match:
        return False

    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3)) if match.group(3) is not None else 0
    return hour < 24 and minute < 60 and second < 60


def is_viable_amount_candidate(clean_text: str, flags: Sequence[str]) -> bool:
    flag_set = set(flags)
    money_evidence = {"money_regex", "target_keyword", "currency_hint", "signed_or_currency"} & flag_set
    if not money_evidence:
        return False

    if {"date_penalty", "colon_time_penalty", "double_colon_penalty"} & flag_set and not (
        {"target_keyword", "currency_hint", "signed_or_currency"} & flag_set
    ):
        return False

    if looks_like_clock_time(clean_text) and "target_keyword" not in flag_set and "signed_or_currency" not in flag_set:
        return False

    if {"masked_account_penalty", "order_keyword_penalty"} & flag_set and "target_keyword" not in flag_set:
        return False

    if "latin_noise_penalty" in flag_set and "target_keyword" not in flag_set and "currency_hint" not in flag_set:
        return False

    return True


def build_fallback_amount_candidates(tokens: Sequence[OCRToken], image_shape: Tuple[int, int, int]) -> List[AmountCandidate]:
    image_h, image_w = image_shape[:2]
    fallbacks: List[AmountCandidate] = []

    for token in tokens:
        clean_text = token.clean_text
        digit_count = len(re.findall(r"\d", clean_text))
        if digit_count < 4:
            continue
        if token.center_y < image_h * 0.12:
            continue
        if DATE_PATTERN.search(clean_text) or looks_like_clock_time(clean_text):
            continue
        if MASKED_ACCOUNT_PATTERN.search(clean_text) or ORDER_PATTERN.fullmatch(clean_text):
            continue
        if token.center_y > image_h * 0.42:
            continue
        if token.width < image_w * 0.12 and token.height < image_h * 0.03:
            continue

        score = 0.35
        flags = ["fallback_digits"]
        if token.width >= image_w * 0.18 or token.height >= image_h * 0.03:
            score += 0.15
            flags.append("prominent")
        if abs(((token.bbox[0] + token.bbox[2]) / 2.0) - image_w / 2.0) <= image_w * 0.22:
            score += 0.1
            flags.append("centered")
        if clean_text.startswith(("+", "-", "¥")):
            score += 0.15
            flags.append("signed_or_currency")

        fallbacks.append(
            AmountCandidate(
                source="fallback",
                text=token.text,
                clean_text=clean_text,
                bbox=token.bbox,
                ocr_confidence=token.conf,
                amount_score=round(score, 4),
                match_flags="|".join(flags),
            )
        )

    return sorted(fallbacks, key=lambda item: (item.amount_score, item.ocr_confidence), reverse=True)


def tokenize_ocr_results(ocr_results: Sequence[Tuple[Sequence[Sequence[float]], str, float]]) -> List[OCRToken]:
    tokens: List[OCRToken] = []
    for bbox, text, conf in ocr_results:
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        clean_text = normalize_text(text)
        if not clean_text:
            continue
        tokens.append(
            OCRToken(
                text=text,
                clean_text=clean_text,
                bbox=(x1, y1, x2, y2),
                conf=float(conf),
                width=width,
                height=height,
                center_y=y1 + height / 2.0,
            )
        )
    return tokens


def group_tokens_by_line(tokens: Sequence[OCRToken]) -> List[List[OCRToken]]:
    groups: List[List[OCRToken]] = []
    for token in sorted(tokens, key=lambda item: item.center_y):
        matched_group: Optional[List[OCRToken]] = None
        for group in groups:
            mean_y = sum(item.center_y for item in group) / len(group)
            mean_h = sum(item.height for item in group) / len(group)
            if abs(token.center_y - mean_y) <= max(token.height, mean_h) * 0.75:
                matched_group = group
                break

        if matched_group is None:
            groups.append([token])
        else:
            matched_group.append(token)

    for group in groups:
        group.sort(key=lambda item: item.bbox[0])
    return groups


def build_amount_candidates(tokens: Sequence[OCRToken], image_shape: Tuple[int, int, int]) -> List[AmountCandidate]:
    candidates: List[AmountCandidate] = []

    for token in tokens:
        score, flags = score_amount_text(token.text, token.bbox, image_shape)
        if score <= 0 or not is_viable_amount_candidate(token.clean_text, flags):
            continue
        candidates.append(
            AmountCandidate(
                source="token",
                text=token.text,
                clean_text=token.clean_text,
                bbox=token.bbox,
                ocr_confidence=token.conf,
                amount_score=score,
                match_flags="|".join(flags),
            )
        )

    for line_tokens in group_tokens_by_line(tokens):
        merged_text = " ".join(token.text for token in line_tokens)
        clean_text = normalize_text(merged_text)
        bbox = (
            min(token.bbox[0] for token in line_tokens),
            min(token.bbox[1] for token in line_tokens),
            max(token.bbox[2] for token in line_tokens),
            max(token.bbox[3] for token in line_tokens),
        )
        mean_conf = float(sum(token.conf for token in line_tokens) / len(line_tokens))
        score, flags = score_amount_text(merged_text, bbox, image_shape)
        if score <= 0 or not is_viable_amount_candidate(clean_text, flags):
            continue
        candidates.append(
            AmountCandidate(
                source="line",
                text=merged_text,
                clean_text=clean_text,
                bbox=bbox,
                ocr_confidence=mean_conf,
                amount_score=score,
                match_flags="|".join(flags),
            )
        )

    deduped: List[AmountCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.amount_score, item.ocr_confidence), reverse=True):
        if any(
            bbox_iou(candidate.bbox, kept.bbox) >= 0.85
            or candidate.clean_text == kept.clean_text
            for kept in deduped
        ):
            continue
        deduped.append(candidate)

    if not deduped:
        deduped = build_fallback_amount_candidates(tokens, image_shape)

    return deduped


def _expanded_candidate_bbox(
    candidate_bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int, int],
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = candidate_bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    image_h, image_w = image_shape[:2]
    return (
        max(0, x1 - int(width * 2.0)),
        max(0, y1 - int(height * 2.0)),
        min(image_w, x2 + int(width * 1.5)),
        min(image_h, y2 + int(height * 2.0)),
    )


def _read_certificate_row_texts(
    image: np.ndarray,
    candidate_bbox: Tuple[int, int, int, int],
    ocr_reader: Any,
) -> Tuple[List[str], Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = _expanded_candidate_bbox(candidate_bbox, image.shape)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return [], (x1, y1, x2, y2)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    row_results = ocr_reader.readtext(enlarged, detail=1, paragraph=False, **CERTIFICATE_ROW_TEXT_OPTIONS)
    row_texts = [normalize_text(item[1]) for item in row_results if item[1].strip()]
    return row_texts, (x1, y1, x2, y2)


def detect_certificate_document_override(
    image_path: Path,
    image: np.ndarray,
    tokens: Sequence[OCRToken],
    candidates: Sequence[AmountCandidate],
    ocr_reader: Any,
) -> Optional[Dict[str, str]]:
    joined_text = " ".join(token.clean_text for token in tokens)
    if not any(keyword in joined_text for keyword in CERTIFICATE_HEADER_KEYWORDS):
        return None

    originality_feats, _, _ = OriginalityChecker.extract_features(str(image_path))
    if not originality_feats:
        return None

    originality_suspicious = (
        originality_feats.get("has_exif", 0) == 0
        and originality_feats.get("size_per_pixel", 0.0) >= 0.18
        and originality_feats.get("color_entropy", 99.0) <= 2.2
    )
    if not originality_suspicious:
        return None

    for candidate in candidates[:3]:
        row_texts, row_bbox = _read_certificate_row_texts(image, candidate.bbox, ocr_reader)
        if not row_texts:
            continue

        row_text = " ".join(row_texts)
        has_amount_row = (
            "交易金额" in row_text
            and "大写" in row_text
            and ("小写" in row_text or "小:" in row_text or "小：" in row_text)
        )
        has_uppercase_amount = bool(re.search(r"大写[:：]?[壹贰叁肆伍陆柒捌玖拾佰仟万零圆整元]+", row_text))
        broken_small_amount = bool(re.search(r"(?:小写|小)[:：]?[¥￥]?\d{5,}元", row_text))
        has_decimal_amount = bool(AMOUNT_PATTERN.search(row_text))
        low_quality_candidate = candidate.ocr_confidence < 0.35 or "." not in candidate.clean_text

        if not (has_amount_row and has_uppercase_amount and broken_small_amount and low_quality_candidate):
            continue
        if has_decimal_amount:
            continue

        confidence = 0.76 if originality_feats.get("color_entropy", 99.0) <= 1.8 else 0.72
        small_amount_match = re.search(r"(?:小写|小)[:：]?([¥￥]?\d{5,}元)", row_text)
        top_text = small_amount_match.group(1) if small_amount_match else candidate.clean_text
        review_priority = 20.0 + confidence * 5.0 + candidate.amount_score
        return {
            "status": "篡改",
            "confidence": f"{confidence:.4f}",
            "reason": CERTIFICATE_RULE_REASON,
            "raw_status": "篡改",
            "raw_confidence": f"{confidence:.4f}",
            "raw_reason": CERTIFICATE_RULE_REASON,
            "text": top_text,
            "bbox": ",".join(str(value) for value in row_bbox),
            "amount_score": f"{candidate.amount_score:.4f}",
            "ocr_confidence": f"{candidate.ocr_confidence:.4f}",
            "source": "document_rule",
            "flags": f"{candidate.match_flags}|{CERTIFICATE_RULE_FLAGS}",
            "review_priority": f"{review_priority:.4f}",
            "has_local_reason": "yes",
            "readable_amount": "no",
            "downgraded": "no",
        }

    return None
