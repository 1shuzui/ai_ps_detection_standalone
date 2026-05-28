import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.ocr_utils import (
    AMOUNT_PATTERN,
    CERTIFICATE_HEADER_KEYWORDS,
    OCRToken,
    AmountCandidate,
    bbox_iou,
    build_amount_candidates,
    detect_certificate_document_override,
    normalize_text,
    tokenize_ocr_results,
)
from core.utils import load_chinese_font, safe_read_image
from inference_api import InferenceEngineAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LOCAL_REASON_KEYWORDS = ("局部字体风格异常", "像素", "高度突变", "基线")
STATUS_RANK = {"正常": 0, "可疑": 1, "篡改": 2, "错误": -1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate amount tamper risk on a folder of receipt images.")
    parser.add_argument("--input-dir", default="ptest", help="Folder containing images to evaluate.")
    parser.add_argument("--output-dir", default="results/amount_batch", help="Folder to write CSV and preview outputs.")
    parser.add_argument("--config", default="config.yaml", help="Inference config path.")
    parser.add_argument("--top-k", type=int, default=4, help="Max amount candidates to evaluate per image.")
    parser.add_argument("--review-threshold", type=float, default=0.35, help="Confidence threshold for preview export.")
    parser.add_argument("--model-version", default=None, help="Specific model version timestamp to use for evaluation.")
    return parser.parse_args()


def load_font(size: int = 18) -> ImageFont.ImageFont:
    return load_chinese_font(size)


def should_export_preview(top_result: Dict[str, str], review_threshold: float) -> bool:
    status = top_result.get("status", "正常")
    confidence = float(top_result.get("confidence", 0.0))
    readable_amount = top_result.get("readable_amount", "no") == "yes"
    downgraded = top_result.get("downgraded", "no") == "yes"
    return status != "正常" or confidence >= review_threshold or top_result.get("candidate_count") == 0 or not readable_amount or downgraded


def is_readable_amount_candidate(candidate_result: Dict[str, str]) -> bool:
    flags = set(filter(None, candidate_result.get("flags", "").split("|")))
    ocr_confidence = float(candidate_result.get("ocr_confidence", 0.0))
    clean_text = candidate_result.get("text", "")
    digit_count = len(re.findall(r"\d", clean_text))
    has_money_shape = "money_regex" in flags or ("target_keyword" in flags and digit_count >= 4)
    if "fallback_digits" in flags and digit_count >= 4:
        has_money_shape = True
    return has_money_shape and ocr_confidence >= 0.2


def draw_preview(
    image_path: Path,
    preview_path: Path,
    top_candidates: Sequence[Dict[str, str]],
    title: str,
) -> None:
    image = safe_read_image(str(image_path))
    if image is None:
        return

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = load_font(20)
    title_font = load_font(22)

    draw.rectangle([(0, 0), (image_pil.width, 34)], fill=(20, 20, 20))
    draw.text((8, 6), title, font=title_font, fill=(255, 255, 255))

    for index, item in enumerate(top_candidates, start=1):
        x1, y1, x2, y2 = [int(value) for value in item["bbox"].split(",")]
        status = item["status"]
        confidence = float(item["confidence"])
        if status == "篡改":
            color = (255, 0, 0)
            text_color = (255, 255, 255)
        elif status == "可疑":
            color = (255, 165, 0)
            text_color = (0, 0, 0)
        else:
            color = (0, 170, 255)
            text_color = (0, 0, 0)

        label = f"{index}. {status} {confidence:.1%} | {item['text'][:20]}"
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        label_h = text_bbox[3] - text_bbox[1]
        label_w = text_bbox[2] - text_bbox[0]
        label_y = max(36, y1 - label_h - 6)
        draw.rectangle([(x1, label_y), (min(image_pil.width, x1 + label_w + 8), label_y + label_h + 6)], fill=color)
        draw.text((x1 + 4, label_y + 3), label, font=font, fill=text_color)

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    result_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    cv2.imencode(".jpg", result_cv2)[1].tofile(str(preview_path))


def evaluate_image(
    image_path: Path,
    engine: InferenceEngineAPI,
    top_k: int,
    review_threshold: float,
    preview_dir: Path,
) -> Dict[str, str]:
    image = safe_read_image(str(image_path))
    if image is None:
        return {
            "image_name": image_path.name,
            "group": image_path.stem.split("_", 1)[0],
            "candidate_count": 0,
            "top_status": "错误",
            "top_confidence": "0.0",
            "top_amount_score": "0.0",
            "top_text": "",
            "top_bbox": "",
            "top_reason": "无法读取图片",
            "preview_path": "",
            "needs_review": "yes",
            "review_priority": "-1.0",
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    ocr_results = engine.extractor.reader.readtext(
        blurred,
        adjust_contrast=0.5,
        mag_ratio=2.0,
        text_threshold=0.25,
    )

    tokens = tokenize_ocr_results(ocr_results)
    candidates = build_amount_candidates(tokens, image.shape)
    shortlisted = candidates[:top_k]
    evaluated: List[Dict[str, str]] = []
    certificate_override = detect_certificate_document_override(
        image_path=image_path,
        image=image,
        tokens=tokens,
        candidates=shortlisted if shortlisted else candidates,
        ocr_reader=engine.extractor.reader,
    )

    for candidate in shortlisted:
        prediction = json.loads(engine.predict(str(image_path), list(candidate.bbox), bbox_format="xyxy"))
        raw_status = prediction.get("result", "错误")
        raw_confidence = float(prediction.get("confidence", 0.0))
        raw_reason = prediction.get("reason", "")
        status = raw_status
        confidence = raw_confidence
        reason = raw_reason
        has_local_reason = any(keyword in reason for keyword in LOCAL_REASON_KEYWORDS)
        has_global_reason = "全局UI布局异常" in reason
        severity = STATUS_RANK.get(status, 0)
        digit_count = len(re.findall(r"\d", candidate.clean_text))
        readable_amount = (
            ("money_regex" in candidate.match_flags or "target_keyword" in candidate.match_flags or "fallback_digits" in candidate.match_flags)
            and digit_count >= 4
            and candidate.ocr_confidence >= 0.2
        )
        flag_set = set(filter(None, candidate.match_flags.split("|")))
        strong_local = (
            has_local_reason
            and readable_amount
            and raw_confidence >= max(0.82, engine.config.get("thresholds", {}).get("suspect_high", 0.65))
            and bool({"target_keyword", "currency_hint", "signed_or_currency", "fallback_digits"} & flag_set)
        )
        downgraded = False

        if raw_status in {"篡改", "可疑"} and not has_global_reason and not strong_local:
            status = "正常"
            confidence = min(raw_confidence, engine.config.get("thresholds", {}).get("suspect_low", 0.50) - 0.01)
            reason = f"局部信号受干扰，转人工复核；原始原因:{raw_reason}"
            downgraded = True
            severity = STATUS_RANK.get(raw_status, 0)

        review_priority = severity * 10 + confidence * 5 + candidate.amount_score
        if has_local_reason:
            review_priority += 0.4
        if has_global_reason and raw_reason == "全局UI布局异常":
            review_priority -= 0.3
        if not readable_amount:
            review_priority += 1.0
        if downgraded:
            review_priority += 0.8

        evaluated.append(
            {
                "status": status,
                "confidence": f"{confidence:.4f}",
                "reason": reason,
                "raw_status": raw_status,
                "raw_confidence": f"{raw_confidence:.4f}",
                "raw_reason": raw_reason,
                "text": candidate.clean_text,
                "bbox": ",".join(str(value) for value in candidate.bbox),
                "amount_score": f"{candidate.amount_score:.4f}",
                "ocr_confidence": f"{candidate.ocr_confidence:.4f}",
                "source": candidate.source,
                "flags": candidate.match_flags,
                "review_priority": f"{review_priority:.4f}",
                "has_local_reason": "yes" if has_local_reason else "no",
                "readable_amount": "yes" if readable_amount else "no",
                "downgraded": "yes" if downgraded else "no",
            }
        )

    if certificate_override and not any(item["status"] == "篡改" for item in evaluated):
        evaluated.append(certificate_override)

    evaluated.sort(
        key=lambda item: (
            STATUS_RANK.get(item["status"], 0),
            float(item["confidence"]),
            float(item["amount_score"]),
            float(item["ocr_confidence"]),
        ),
        reverse=True,
    )

    preview_path = ""
    if evaluated:
        top_result = evaluated[0]
        top_result["candidate_count"] = len(evaluated)
        if should_export_preview(
            {
                    "status": top_result["status"],
                    "confidence": top_result["confidence"],
                    "candidate_count": len(evaluated),
                    "readable_amount": "yes" if is_readable_amount_candidate(top_result) else "no",
                    "downgraded": top_result["downgraded"],
                },
                review_threshold,
            ):
            preview_path = str(preview_dir / image_path.with_suffix(".jpg").name)
            draw_preview(
                image_path=image_path,
                preview_path=Path(preview_path),
                top_candidates=evaluated[:3],
                title=f"{image_path.name} | {top_result['status']} {float(top_result['confidence']):.1%}",
            )

        return {
            "image_name": image_path.name,
            "group": image_path.stem.split("_", 1)[0],
            "candidate_count": str(len(evaluated)),
            "top_status": top_result["status"],
            "top_confidence": top_result["confidence"],
            "top_amount_score": top_result["amount_score"],
            "top_text": top_result["text"],
            "top_bbox": top_result["bbox"],
            "top_reason": top_result["reason"],
            "top_raw_status": top_result["raw_status"],
            "top_raw_confidence": top_result["raw_confidence"],
            "top_raw_reason": top_result["raw_reason"],
            "top_source": top_result["source"],
            "top_flags": top_result["flags"],
            "top_ocr_confidence": top_result["ocr_confidence"],
            "top_has_local_reason": top_result["has_local_reason"],
            "top_readable_amount": "yes" if is_readable_amount_candidate(top_result) else "no",
            "top_downgraded": top_result["downgraded"],
            "review_priority": top_result["review_priority"],
            "preview_path": preview_path,
            "needs_review": "yes"
            if should_export_preview(
                {
                    "status": top_result["status"],
                    "confidence": top_result["confidence"],
                    "candidate_count": len(evaluated),
                    "readable_amount": "yes" if is_readable_amount_candidate(top_result) else "no",
                    "downgraded": top_result["downgraded"],
                },
                review_threshold,
            )
            else "no",
            "top3_candidates_json": json.dumps(evaluated[:3], ensure_ascii=False),
        }

    return {
        "image_name": image_path.name,
        "group": image_path.stem.split("_", 1)[0],
        "candidate_count": "0",
        "top_status": "未定位",
        "top_confidence": "0.0",
        "top_amount_score": "0.0",
        "top_text": "",
        "top_bbox": "",
        "top_reason": "未找到金额候选区域",
        "top_raw_status": "未定位",
        "top_raw_confidence": "0.0",
        "top_raw_reason": "未找到金额候选区域",
        "top_source": "",
        "top_flags": "",
        "top_ocr_confidence": "0.0",
        "top_has_local_reason": "no",
        "top_readable_amount": "no",
        "top_downgraded": "no",
        "review_priority": "-0.5",
        "preview_path": "",
        "needs_review": "yes",
        "top3_candidates_json": "[]",
    }


def write_csv(rows: Sequence[Dict[str, str]], csv_path: Path) -> None:
    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_run_summary(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    summary = {
        "total_images": len(rows),
        "tampered": 0,
        "suspicious": 0,
        "normal": 0,
        "unlocated": 0,
        "needs_review": 0,
        "by_group": {},
    }

    for row in rows:
        status = row["top_status"]
        group = row["group"]
        summary["by_group"].setdefault(group, {"total": 0, "needs_review": 0, "tampered": 0, "suspicious": 0})
        summary["by_group"][group]["total"] += 1

        if row["needs_review"] == "yes":
            summary["needs_review"] += 1
            summary["by_group"][group]["needs_review"] += 1

        if status == "篡改":
            summary["tampered"] += 1
            summary["by_group"][group]["tampered"] += 1
        elif status == "可疑":
            summary["suspicious"] += 1
            summary["by_group"][group]["suspicious"] += 1
        elif status == "未定位":
            summary["unlocated"] += 1
        else:
            summary["normal"] += 1

    return summary


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "review_previews"

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in valid_extensions)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    logger.info("加载金额专用批评估引擎...")
    engine = InferenceEngineAPI(args.config)
    if args.model_version:
        logger.info("切换到指定模型版本: %s", args.model_version)
        engine.reload_models(version=args.model_version)

    rows: List[Dict[str, str]] = []
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        row = evaluate_image(
            image_path=image_path,
            engine=engine,
            top_k=args.top_k,
            review_threshold=args.review_threshold,
            preview_dir=preview_dir,
        )
        rows.append(row)
        if index == 1 or index % 25 == 0 or index == total:
            logger.info(
                "进度 %s/%s | %s | %s %.1f%% | candidates=%s",
                index,
                total,
                image_path.name,
                row["top_status"],
                float(row["top_confidence"]) * 100,
                row["candidate_count"],
            )

    rows_sorted = sorted(
        rows,
        key=lambda item: (
            item["needs_review"] == "yes",
            float(item["review_priority"]),
            STATUS_RANK.get(item["top_status"], -1),
            float(item["top_confidence"]),
        ),
        reverse=True,
    )
    review_rows = [row for row in rows_sorted if row["needs_review"] == "yes"]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows_sorted, output_dir / "amount_batch_summary.csv")
    write_csv(review_rows, output_dir / "amount_batch_review.csv")

    summary = build_run_summary(rows_sorted)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    logger.info("批评估完成。summary=%s review=%s", output_dir / "amount_batch_summary.csv", output_dir / "amount_batch_review.csv")
    logger.info("统计摘要: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
