import os
import json
import numpy as np
import cv2
import xgboost as xgb
import joblib
import logging
import re
from core.augmentations import build_global_augmentations, build_roi_augmentations
from core.extractors import FeatureExtractor, FontFeatureLibrary
from core.utils import safe_read_image

# 配置标准日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_pipeline():
    logger.info("启动双流架构离线训练流水线")
    extractor = FeatureExtractor()
    font_lib = FontFeatureLibrary()

    json_dir = "locate_json"
    images_dir = "images"

    # --- 1. 训练局部字体库 ---
    if os.path.exists(json_dir):
        logger.info("开始构建 FAISS 字体特征库...")
        font_augmented_regions = 0
        for fname in os.listdir(json_dir):
            if not fname.endswith('.json') or 'no' not in fname.lower():
                continue

            with open(os.path.join(json_dir, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)

            raw_image_path = str(data.get('image_path', '')).replace('\\', '/')
            image_name = os.path.basename(raw_image_path)
            img_path = os.path.join(images_dir, image_name)
            img = safe_read_image(img_path)
            if img is None: continue

            seen_regions = set()
            for region in data.get('key_regions', []):
                if region.get('type') == 'amount' and 'bbox' in region:
                    region_text = str(region.get('text', '')).replace(' ', '')
                    if len(re.findall(r'\d', region_text)) < 3:
                        continue

                    bbox = [int(v) for v in region['bbox']]
                    key = (tuple(bbox), region_text, region.get('type', ''))
                    if key in seen_regions:
                        continue
                    seen_regions.add(key)

                    x1, y1, x2, y2 = bbox
                    roi = img[y1:y2, x1:x2]

                    if roi.size > 0:
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        roi_variants = [("orig", roi_rgb)]
                        roi_variants.extend(build_roi_augmentations(roi_rgb, f"{fname}:{region_text}:{bbox}"))

                        for variant_name, variant_img in roi_variants:
                            feats, _stats, feature_texts = extractor.extract_from_roi(variant_img)
                            if feats:
                                font_lib.add(feats, feature_texts)
                            if variant_name != "orig":
                                font_augmented_regions += 1

        font_lib.save("models/font_lib")
        logger.info(f"FAISS 库更新完毕，当前特征总数: {font_lib.index.ntotal} | 增强ROI样本: {font_augmented_regions}")

    # --- 2. 训练全局深度分类器 ---
    logger.info("开始提取全局深层 UI 特征 (遍历所有样本)...")
    X_global, y_global = [], []
    global_augmented_images = 0

    if os.path.exists(images_dir):
        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

            img_path = os.path.join(images_dir, img_name)
            img = safe_read_image(img_path)
            if img is None: continue

            # 自动打标签
            if 'no' in img_name.lower():
                label = 0
            elif 'p' in img_name.lower():
                label = 1
            else:
                continue

            global_variants = [("orig", img)]
            global_variants.extend(build_global_augmentations(img, img_name))

            for variant_name, variant_img in global_variants:
                global_feat = extractor.extract_global_feature(variant_img)
                X_global.append(global_feat)
                y_global.append(label)
                if variant_name != "orig":
                    global_augmented_images += 1

        logger.info(f"共提取 {len(X_global)} 张图片的全局特征 (其中假图样本 {sum(y_global)} 张, 增强样本 {global_augmented_images} 张)")

        if len(X_global) > 0:
            logger.info("正在基于 512维深层特征拟合 XGBoost 模型...")
            model = xgb.XGBClassifier(eval_metric='logloss', max_depth=6, n_estimators=150, random_state=42)
            model.fit(np.array(X_global), np.array(y_global))

            save_path = "models/global_layout_model.pkl"
            joblib.dump(model, save_path)
            logger.info(f"全局大模型训练并保存成功，路径: {save_path}")

    logger.info("训练流水线执行结束")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_pipeline()
