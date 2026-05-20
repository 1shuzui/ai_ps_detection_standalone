import os
import cv2
import json
import numpy as np
import easyocr
import re
import torch
import logging
from PIL import Image, ImageDraw, ImageFont
from batch_eval_amounts import build_amount_candidates, tokenize_ocr_results
from core.utils import load_chinese_font
from inference_api import InferenceEngineAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_raw_image(image_path, engine, ocr_reader, output_dir="results/new_tests"):
    logger.info(f"正在处理图片: {os.path.basename(image_path)}")

    img_cv2 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_cv2 is None:
        logger.error(f"无法读取图片: {image_path}")
        return

    logger.info("执行 OCR 全图扫描定位 (启用中值滤波抗摩尔纹)...")

    gray_for_ocr = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    # 【核心修正】：放弃 CLAHE（会放大网格），改用 3x3 中值滤波。
    # 中值滤波是消除屏幕像素点阵、椒盐噪声的最强武器。
    blurred_for_ocr = cv2.medianBlur(gray_for_ocr, 3)

    # mag_ratio=2.0：将图像内部放大2倍，配合低text_threshold，强行把金额从模糊中抠出来
    ocr_results = ocr_reader.readtext(
        blurred_for_ocr,
        adjust_contrast=0.5,
        mag_ratio=2.0,
        text_threshold=0.25
    )

    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = load_chinese_font(22)

    tampered_count = 0
    checked_count = 0

    logger.info("调用 AI 引擎执行鉴伪核查...")

    #     # 捕获核心金额特征 (支持千分位与两位小数，如 55,589.00 或 55589.00)
    #     is_amount = bool(re.search(r'\d{1,3}(,\d{3})*\.\d{2}', text)) or bool(re.search(r'\d+\.\d{2}', text))
    #     # 捕获核心时间特征 (如 2026-03-16 或 2026/03/16)
    #     is_date = bool(re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', text))

    candidates = build_amount_candidates(tokenize_ocr_results(ocr_results), img_cv2.shape)

    # 3. 遍历筛出的金额候选框
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.bbox
        roi_bbox = [x1, y1, x2, y2]

        checked_count += 1

        ai_result_str = engine.predict(image_path, roi_bbox, bbox_format="xyxy")
        ai_result_dict = json.loads(ai_result_str)

        real_status = ai_result_dict.get("result", "正常")
        confidence = ai_result_dict.get("confidence", 0.0)

        if real_status == "篡改":
            color = (255, 0, 0)
            text_color = (255, 255, 255)
            label = f"篡改 | 风险:{confidence:.1%}"
            tampered_count += 1
        elif real_status == "可疑":
            color = (255, 165, 0)
            text_color = (0, 0, 0)
            label = f"可疑 | 风险:{confidence:.1%}"
            tampered_count += 1
        else:
            color = (0, 255, 0)
            text_color = (0, 0, 0)
            label = f"正常 | 风险:{confidence:.1%}"

        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        label_bg_y1 = max(y1 - text_height - 6, 0)
        draw.rectangle([(x1, label_bg_y1), (min(x1 + text_width + 6, img_pil.width), max(y1, text_height + 6))], fill=color)
        draw.text((x1 + 3, label_bg_y1 + 3), label, font=font, fill=text_color)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"BlindTest_{os.path.basename(image_path)}")
    img_cv2_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imencode('.jpg', img_cv2_result)[1].tofile(save_path)

    logger.info(f"分析完毕: 核验 {checked_count} 处关键数值，发现 {tampered_count} 处异常。")
    logger.info(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    logger.info("启动图片端到端 (E2E) 盲测程序")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"加载 OCR 定位组件 (硬件: {device})...")
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=(device == 'cuda'))

    logger.info("加载 AI 鉴伪核心引擎...")
    engine = InferenceEngineAPI("config.yaml")

    test_folder = "ptest"
    os.makedirs(test_folder, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    new_images = [f for f in os.listdir(test_folder) if f.lower().endswith(valid_extensions)]

    if not new_images:
        logger.warning(f"请将待测试的新图片放入 '{test_folder}' 文件夹中。")
    else:
        for img_name in new_images:
            img_path = os.path.join(test_folder, img_name)
            test_raw_image(img_path, engine, reader)

        logger.info("所有图片测试完毕。")
