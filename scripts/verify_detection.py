"""
验证脚本：对 images/ + pptest/ 全部图片运行检测，输出结果到 results/ 目录。

用法: python scripts/verify_detection.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
os.chdir(str(_project_root))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inference_api import InferenceEngineAPI
from core.utils import load_chinese_font


def draw_visualization(img_path: str, result_data: dict, output_path: str) -> None:
    """在图片上绘制检测框和标签，保存到 output_path。"""
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = load_chinese_font(20)
    font_sm = load_chinese_font(14)

    h, w = img.shape[:2]
    result_label = result_data.get("result", "?")
    confidence = result_data.get("confidence", 0)
    bbox = result_data.get("bbox", [0, 0, 0, 0])
    reasons = result_data.get("reason", "")

    # 颜色映射
    colors = {"篡改": (220, 38, 38), "可疑": (234, 179, 8), "正常": (34, 197, 94)}
    color = colors.get(result_label, (100, 100, 100))

    # 绘制 ROI 框
    x, y, rw, rh = [int(v) for v in bbox]
    if rw > 0 and rh > 0:
        draw.rectangle([x, y, x + rw, y + rh], outline=color, width=3)

    # 顶部标签条
    label_text = f"{result_label}  {confidence * 100:.1f}%"
    bar_h = 36
    draw.rectangle([(0, 0), (w, bar_h)], fill=(*color, 200))
    try:
        draw.text((10, 6), label_text, fill=(255, 255, 255), font=font)
    except Exception:
        draw.text((10, 6), label_text, fill=(255, 255, 255))

    # 底部理由（截断）
    if reasons:
        reason_short = reasons[:80] + ("..." if len(reasons) > 80 else "")
        bar_y = max(0, h - 28)
        draw.rectangle([(0, bar_y), (w, h)], fill=(0, 0, 0, 180))
        try:
            draw.text((8, bar_y + 4), reason_short, fill=(220, 220, 220), font=font_sm)
        except Exception:
            draw.text((8, bar_y + 4), reason_short, fill=(220, 220, 220))

    img_pil.save(output_path, "JPEG", quality=90)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"verification_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    engine = InferenceEngineAPI(config_path="config.yaml")
    images_dir = Path("images")
    pptest_dir = Path("pptest")

    lines = []
    def w(s=""):
        lines.append(s)

    # ---- 头部 ----
    w("=" * 70)
    w("AI 图像鉴伪检测 — 全量验证报告")
    w(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"模型路径: {engine._xgb_path}")
    w(f"字体库: {engine.font_lib.index.ntotal} 条, ready={engine.font_lib.is_ready}")
    w(f"温度校准: T={engine._calibration_temp}")
    w(f"融合权重: global={engine.config['fusion']['weight_global']}, "
        f"local={engine.config['fusion']['weight_local']}")
    w(f"阈值: suspect_low={engine.config['thresholds']['suspect_low']}, "
        f"suspect_high={engine.config['thresholds']['suspect_high']}")
    w("=" * 70)
    w()

    # ---- 运行检测 + 生成可视化 ----
    all_results = {"no": [], "p": [], "pptest": []}
    locate_dir = Path("locate_json")

    def get_roi_from_locate_json(fname: str, img_w: int, img_h: int):
        """从 locate_json 中取面积最大的关键区域作为检测 ROI。"""
        json_path = locate_dir / fname.replace(".jpg", ".json").replace(".png", ".json")
        if not json_path.exists():
            # 回退：全图中心区域
            cx, cy = img_w // 2, img_h // 2
            return [max(0, cx - 150), max(0, cy - 50), min(300, img_w), min(100, img_h)]
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                anno = json.load(jf)
            regions = anno.get("key_regions", [])
            if not regions:
                cx, cy = img_w // 2, img_h // 2
                return [max(0, cx - 150), max(0, cy - 50), min(300, img_w), min(100, img_h)]
            # 取面积最大的区域
            best = max(regions, key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]))
            bbox = best["bbox"]  # [x1, y1, x2, y2]
            return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        except Exception:
            cx, cy = img_w // 2, img_h // 2
            return [max(0, cx - 150), max(0, cy - 50), min(300, img_w), min(100, img_h)]

    for prefix, label, src_dir, ext in [
        ("no", "NORMAL", images_dir, ".jpg"),
        ("p", "TAMPERED", images_dir, ".jpg"),
        ("pptest", "PPTEST", pptest_dir, ".png"),
    ]:
        if prefix == "pptest":
            files = sorted([f for f in os.listdir(src_dir) if f.endswith(ext)])
        else:
            files = sorted([f for f in os.listdir(src_dir)
                            if f.startswith(prefix + " ") and f.endswith(ext)
                            and "enhanced" not in f])

        for fname in files:
            img_path = str(src_dir / fname)

            # 读取图片尺寸，从 locate_json 获取 ROI
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                ih, iw = img.shape[:2]
                roi_bbox = get_roi_from_locate_json(fname, iw, ih)
            else:
                roi_bbox = [50, 50, 300, 100]

            result_json = engine.predict(full_image_path=img_path, roi_bbox=roi_bbox)
            data = json.loads(result_json)
            all_results[prefix].append((data, fname))

            # 生成可视化图
            viz_name = f"{Path(fname).stem}_viz.jpg"
            viz_path = str(viz_dir / viz_name)
            try:
                draw_visualization(img_path, data, viz_path)
            except Exception:
                pass

    # ---- 统计输出 ----
    total_ok = 0
    total_all = 0

    for category, entries, expected_ok in [
        ("正常图片 (no)", all_results["no"], lambda d: d["result"] == "正常"),
        ("篡改图片 (p)", all_results["p"], lambda d: d["result"] == "篡改"),
        ("测试图片 (pptest)", all_results["pptest"], lambda d: d["result"] == "篡改"),
    ]:
        ok = sum(1 for d, _ in entries if expected_ok(d))
        total = len(entries)
        total_ok += ok
        total_all += total
        pct = ok / total * 100 if total else 0
        status = "✓ PASS" if pct == 100 else f"✗ FAIL ({total - ok} errors)"

        confs = sorted([d["confidence"] for d, _ in entries])

        w(f"[{category}] {ok}/{total} = {pct:.1f}%  {status}")
        w(f"  置信度范围: [{confs[0]:.4f}, {confs[-1]:.4f}]  "
          f"中位数: {confs[len(confs)//2]:.4f}")
        w("-" * 50)

        for data, fname in entries:
            actual = data["result"]
            conf = data["confidence"]
            reason = data.get("reason", "")
            is_ok = expected_ok(data)
            mark = "✓" if is_ok else "✗"
            viz_file = f"{Path(fname).stem}_viz.jpg"
            w(f"  {mark} {fname:30s} {actual:4s}  conf={conf:.4f}  viz={viz_file}")
            if reason and reason != "未检出明显篡改痕迹":
                w(f"    理由: {reason}")
        w()

    # ---- 总结 ----
    overall = total_ok / total_all * 100 if total_all else 0
    w("=" * 70)
    w(f"总计: {total_ok}/{total_all} = {overall:.1f}%")
    w(f"状态: {'✓ ALL PASS' if overall == 100 else '✗ FAIL'}")
    w(f"可视化图片: {viz_dir}")
    w(f"报告保存于: {results_dir}")
    w("=" * 70)

    # ---- 写入文件 + 控制台 ----
    text = "\n".join(lines)
    report_path = results_dir / "report.txt"
    report_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n报告: {report_path}")
    print(f"可视化: {viz_dir}/")


if __name__ == "__main__":
    main()
