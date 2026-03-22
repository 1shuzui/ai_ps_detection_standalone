import cv2
import json
import yaml
import numpy as np
import logging
import os
import joblib
import re
from typing import List

from core.extractors import FeatureExtractor, FontFeatureLibrary, TamperAnalyzer
from core.detectors import PixelLevelDetector
from core.utils import NumpyEncoder, safe_read_image

# 配置标准日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceEngineAPI:
    def __init__(self, config_path="config.yaml"):
        # 引擎初始化时读取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.extractor = FeatureExtractor()
        self.font_lib = FontFeatureLibrary()
        self.font_lib.load(self.config['paths']['font_lib_path'])

        # 从配置中读取全局模型路径（兼容缺省路径）
        xgb_path = self.config.get('paths', {}).get('xgb_model_path', "models/global_layout_model.pkl")
        self.global_model = joblib.load(xgb_path)
        self.pixel_detector = PixelLevelDetector()

    def predict(self, full_image_path: str, roi_bbox: List[int]) -> str:
        # 【终极防御】用 Try-Except 包裹，防止任何内部错误导致后端服务崩溃
        try:
            reasons = []
            result_status = "正常"

            # 【路径兼容】使用安全读取函数，彻底解决 cv2.imread 无法读取中文路径的问题
            img = safe_read_image(full_image_path)
            if img is None:
                return json.dumps({"result": "错误", "reason": "无法读取图片或路径不存在"}, ensure_ascii=False)

            img_h, img_w = img.shape[:2]

            # ================== 动态读取配置 (告别魔法数字) ==================
            rules = self.config.get('business_rules', {})
            weights = self.config.get('weights', {})
            thresh = self.config.get('thresholds', {})

            margin = rules.get('roi_expand_margin', 15)
            max_len = rules.get('max_core_text_length', 15)

            thresh_global = thresh.get('global_fake', 0.65)
            thresh_pixel_alert = thresh.get('pixel_anomaly_alert', 0.60)
            thresh_exempt = thresh.get('exempt_pixel_safe', 0.40)
            thresh_high = thresh.get('suspect_high', 0.65)
            thresh_low = thresh.get('suspect_low', 0.50)

            # ================== BBox 严密越界保护 ==================
            raw_x, raw_y = roi_bbox[0], roi_bbox[1]
            raw_w = roi_bbox[2] if roi_bbox[2] < 2000 else roi_bbox[2] - raw_x
            raw_h = roi_bbox[3] if roi_bbox[3] < 2000 else roi_bbox[3] - raw_y

            # 强制限制在图片物理尺寸内，防止切片时越界报错
            x1 = max(0, min(raw_x, img_w - 1))
            y1 = max(0, min(raw_y, img_h - 1))
            x2 = max(x1 + 1, min(raw_x + raw_w, img_w))
            y2 = max(y1 + 1, min(raw_y + raw_h, img_h))

            x, y = x1, y1
            w, h = x2 - x1, y2 - y1

            # ================== 1. 全局特征分析 ==================
            global_feat = self.extractor.extract_global_feature(img)
            global_fake_prob = float(self.global_model.predict_proba(np.array([global_feat]))[0][1])

            # ================== 2. 局部微观分析 ==================
            # 对外扩区域同样做越界保护
            x_exp, y_exp = max(0, x - margin), max(0, y - margin)
            w_exp = min(img_w - x_exp, w + 2 * margin)
            h_exp = min(img_h - y_exp, h + 2 * margin)

            roi_img = img[y:y + h, x:x + w]
            roi_img_expanded = img[y_exp:y_exp + h_exp, x_exp:x_exp + w_exp]

            roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            feats, stats = self.extractor.extract_from_roi(roi_rgb)

            extracted_text = "".join([s['text'] for s in stats])

            font_sim = np.mean([self.font_lib.search_similarity(f) for f in feats]) if feats else 0.5
            font_anomaly = max(0.0, 1.0 - font_sim)

            pixel_anomaly = self.pixel_detector.detect(roi_img_expanded)
            geo_reasons, geo_penalty = TamperAnalyzer.check_internal_consistency(stats)

            # ================== 3. 自适应权重计算 ==================
            is_non_core_amount = bool(re.search(r'[\u4e00-\u9fa5a-zA-Z]', extracted_text)) or len(
                extracted_text) > max_len

            if is_non_core_amount or len(extracted_text) == 0:
                # 非核心字段豁免逻辑
                local_tamper_prob = pixel_anomaly * weights.get('non_core_pixel', 0.8)
                geo_penalty = 0.0
                if pixel_anomaly < thresh_exempt:
                    local_tamper_prob = 0.0
            else:
                # 核心字段双重校验逻辑
                local_tamper_prob = (pixel_anomaly * weights.get('core_pixel', 0.6)) + (
                            font_anomaly * weights.get('core_font', 0.4)) + geo_penalty

            final_risk = max(global_fake_prob, local_tamper_prob)
            final_risk = max(0.0, min(1.0, float(final_risk)))

            # ================== 4. 结果判定与防篡改理由梳理 ==================
            if global_fake_prob > thresh_global:
                reasons.append("全局UI布局异常")
            if pixel_anomaly > thresh_pixel_alert:
                reasons.append("存在局部边缘拼接/像素涂抹痕迹")
            if geo_penalty > 0:
                reasons.extend(geo_reasons)

            if final_risk > thresh_high:
                result_status = "篡改"
            elif final_risk > thresh_low:
                result_status = "可疑"
            else:
                if not reasons:
                    reasons.append("未检出明显篡改痕迹")

            output = {
                "result": result_status,
                "confidence": final_risk,
                "bbox": [int(i) for i in [x, y, w, h]],
                "reason": "；".join(reasons)
            }
            return json.dumps(output, ensure_ascii=False, indent=4, cls=NumpyEncoder)

        except Exception as e:
            # 捕获所有未知的严重错误，并标准格式化返回
            logger.error(f"引擎推理引发未捕获异常: {e}", exc_info=True)
            error_output = {
                "result": "错误",
                "confidence": 0.0,
                "bbox": roi_bbox,
                "reason": f"引擎内部解析失败: {str(e)}"
            }
            return json.dumps(error_output, ensure_ascii=False, indent=4, cls=NumpyEncoder)


# =====================================================================
# 下方为本地独立测试代码，当此脚本被直接运行时触发
# =====================================================================
if __name__ == "__main__":
    import time

    logger.info("启动单图推理本地测试 (Inference API)")

    try:
        engine = InferenceEngineAPI(config_path="config.yaml")
        logger.info("引擎初始化成功")
    except Exception as e:
        logger.error(f"引擎初始化失败: {e}", exc_info=True)
        exit(1)

    # 替换为你 images/ 文件夹下真实存在的图片进行本地测试
    test_image_path = "pptest/111.png"
    test_bbox = [150, 200, 180, 45]

    if not os.path.exists(test_image_path):
        logger.warning(f"找不到测试图片: {test_image_path}，请修改路径后重试。")
    else:
        logger.info(f"目标图片: {test_image_path} | BBox: {test_bbox}")
        start_time = time.time()

        try:
            result_json = engine.predict(full_image_path=test_image_path, roi_bbox=test_bbox)
            cost_time = time.time() - start_time
            logger.info(f"推理耗时: {cost_time:.3f} 秒")
            logger.info(f"返回结果:\n{result_json}")
        except Exception as e:
            logger.error(f"推理过程中发生错误: {e}", exc_info=True)