import cv2
import json
import yaml
import numpy as np
import logging
import os
import joblib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.extractors import FeatureExtractor, FontFeatureLibrary, TamperAnalyzer
from core.detectors import PixelLevelDetector, OriginalityChecker
from core.utils import NumpyEncoder, safe_read_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceEngineAPI:
    def __init__(self, config_path="config.yaml", shared_ocr_reader: Optional[Any] = None):
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = (Path(__file__).resolve().parent / config_file).resolve()

        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.base_dir = config_file.parent

        self.extractor = FeatureExtractor(reader=shared_ocr_reader)
        self.font_lib = FontFeatureLibrary()
        font_lib_path = self._resolve_path(self.config['paths']['font_lib_path'])
        self.font_lib.load(font_lib_path)

        xgb_path = self.config.get('paths', {}).get('xgb_model_path', "models/global_layout_model.pkl")
        self.global_model = joblib.load(self._resolve_path(xgb_path))

        pixel_cfg = self.config.get('pixel_detector', {})
        self.pixel_detector = PixelLevelDetector(config=pixel_cfg)

        self.originality_checker = OriginalityChecker()
        self._origin_enabled = self.config.get('originality', {}).get('enabled', True)

    def _resolve_path(self, path_str: str) -> str:
        path = Path(path_str)
        if path.is_absolute():
            return str(path)
        return str((self.base_dir / path).resolve())

    @staticmethod
    def _clip_bbox_xyxy(bbox_xyxy: List[int], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        return x1, y1, x2, y2

    def _normalize_roi_bbox(self, roi_bbox: List[int], img_w: int, img_h: int, bbox_format: str) -> Tuple[int, int, int, int]:
        if len(roi_bbox) != 4:
            raise ValueError("ROI bbox must contain exactly four integers.")

        x1, y1, third, fourth = [int(v) for v in roi_bbox]
        format_name = (bbox_format or "auto").lower()

        if format_name == "xyxy":
            return self._clip_bbox_xyxy([x1, y1, third, fourth], img_w, img_h)

        if format_name == "xywh":
            return self._clip_bbox_xyxy([x1, y1, x1 + third, y1 + fourth], img_w, img_h)

        looks_like_xyxy = third > x1 and fourth > y1 and third <= img_w and fourth <= img_h
        if looks_like_xyxy:
            return self._clip_bbox_xyxy([x1, y1, third, fourth], img_w, img_h)

        return self._clip_bbox_xyxy([x1, y1, x1 + third, y1 + fourth], img_w, img_h)

    @staticmethod
    def _profile_numeric_text(extracted_text: str, max_len: int) -> Dict[str, float]:
        text_clean = extracted_text.replace(" ", "")
        total_len = len(text_clean)
        digit_count = len(re.findall(r"\d", text_clean))
        digit_ratio = (digit_count / total_len) if total_len else 0.0

        amount_pattern = re.search(r"\d[\d,]*[.:]\d{1,2}", text_clean)
        currency_hint = re.search(r"(小写|金额|元|¥|￥|人民币)", text_clean)
        order_hint = re.search(r"(单号|订单|流水|凭证|参考号)", text_clean)

        is_core_candidate = digit_count >= 3 and (
            digit_ratio >= 0.35 or amount_pattern is not None or currency_hint is not None or order_hint is not None
        )
        should_use_font_signal = is_core_candidate or (digit_count >= 3 and total_len <= max_len * 2)

        return {
            "digit_count": digit_count,
            "digit_ratio": digit_ratio,
            "total_len": total_len,
            "is_core_candidate": float(is_core_candidate),
            "should_use_font_signal": float(should_use_font_signal),
        }

    def predict(
        self,
        full_image_path: str,
        roi_bbox: List[int],
        bbox_format: str = "auto",
        precomputed_global_feat: Optional[np.ndarray] = None,
    ) -> str:
        try:
            reasons = []
            result_status = "正常"

            img = safe_read_image(full_image_path)
            if img is None:
                return json.dumps({"result": "错误", "reason": "无法读取图片或路径不存在"}, ensure_ascii=False)

            img_h, img_w = img.shape[:2]

            rules = self.config.get('business_rules', {})
            weights = self.config.get('weights', {})
            thresh = self.config.get('thresholds', {})
            fusion_cfg = self.config.get('fusion', {})

            margin = rules.get('roi_expand_margin', 15)
            max_len = rules.get('max_core_text_length', 15)

            thresh_global = thresh.get('global_fake', 0.65)
            thresh_pixel_alert = thresh.get('pixel_anomaly_alert', 0.60)
            thresh_exempt = thresh.get('exempt_pixel_safe', 0.40)
            thresh_high = thresh.get('suspect_high', 0.65)
            thresh_low = thresh.get('suspect_low', 0.50)

            fusion_method = fusion_cfg.get('method', 'weighted')
            w_global = fusion_cfg.get('weight_global', 0.35)
            w_local = fusion_cfg.get('weight_local', 0.65)

            x1, y1, x2, y2 = self._normalize_roi_bbox(roi_bbox, img_w, img_h, bbox_format)
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            # ================== 0. EXIF/元数据分析 ==================
            metadata_risk = 0.0
            if self._origin_enabled:
                orig_feats, hard_rule, _ = self.originality_checker.extract_features(full_image_path)
                if hard_rule:
                    reasons.append("EXIF检测到已知修图软件")
                    metadata_risk = 0.55
                elif orig_feats:
                    m_risk, m_reasons = OriginalityChecker.compute_metadata_risk(orig_feats)
                    metadata_risk = m_risk
                    reasons.extend(m_reasons)

            # ================== 1. 全局特征分析 ==================
            if precomputed_global_feat is not None:
                global_feat = precomputed_global_feat
            else:
                global_feat = self.extractor.extract_global_feature(img)
            global_fake_prob = float(self.global_model.predict_proba(np.array([global_feat]))[0][1])

            # ================== 2. 局部微观分析 ==================
            x_exp, y_exp = max(0, x - margin), max(0, y - margin)
            w_exp = min(img_w - x_exp, w + 2 * margin)
            h_exp = min(img_h - y_exp, h + 2 * margin)

            roi_img = img[y:y + h, x:x + w]
            roi_img_expanded = img[y_exp:y_exp + h_exp, x_exp:x_exp + w_exp]

            roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            feats, stats = self.extractor.extract_from_roi(roi_rgb)

            feature_texts = [s['text'] for s in stats if s.get('is_core_number')]
            extracted_text = "".join(feature_texts) if feature_texts else "".join([s['text'] for s in stats])
            text_profile = self._profile_numeric_text(extracted_text, max_len)
            should_use_font_signal = bool(text_profile["should_use_font_signal"])

            # 批量字体相似度查询
            font_sims = self.font_lib.search_similarity_batch(feats) if feats else []
            font_sim = np.mean(font_sims) if font_sims else 0.5
            font_anomaly = max(0.0, 1.0 - font_sim)

            # 像素检测（增加周围背景用于噪声一致性对比）
            surrounding = None
            if margin > 0 and img.size > 0:
                sur_x1 = max(0, x - margin * 4)
                sur_y1 = max(0, y - margin * 4)
                sur_x2 = min(img_w, x + w + margin * 4)
                sur_y2 = min(img_h, y + h + margin * 4)
                surrounding = img[sur_y1:sur_y2, sur_x1:sur_x2]

            pixel_anomaly = self.pixel_detector.detect(roi_img_expanded, surrounding_np=surrounding)
            geo_reasons, geo_penalty = TamperAnalyzer.check_internal_consistency(stats)

            # ================== 3. 自适应权重计算 ==================
            if should_use_font_signal and len(extracted_text) > 0:
                local_tamper_prob = (
                    pixel_anomaly * weights.get('core_pixel', 0.6)
                    + font_anomaly * weights.get('core_font', 0.4)
                    + geo_penalty
                )
                if text_profile["digit_count"] >= 8 and font_anomaly > 0.75:
                    local_tamper_prob = max(local_tamper_prob, thresh_low + 0.02)
            else:
                local_tamper_prob = (pixel_anomaly * weights.get('non_core_pixel', 0.8)) + geo_penalty
                if pixel_anomaly < thresh_exempt and geo_penalty == 0:
                    local_tamper_prob = 0.0

            local_tamper_prob = max(0.0, min(1.0, float(local_tamper_prob)))

            # ================== 4. 融合策略（max + 一致信号加成） ==================
            # 基准：取最强信号
            final_risk = max(global_fake_prob, local_tamper_prob, metadata_risk)

            if fusion_method == "max":
                pass  # 直接使用 max 结果
            else:
                # 加权融合：max 保底 + 多信号一致时加成
                signals_above_threshold = 0
                if global_fake_prob > thresh_global * 0.8:
                    signals_above_threshold += 1
                if local_tamper_prob > thresh_low * 0.9:
                    signals_above_threshold += 1
                if metadata_risk > 0.4:
                    signals_above_threshold += 1

                if signals_above_threshold >= 2:
                    agreement_bonus = 0.08 if signals_above_threshold == 2 else 0.12
                    final_risk = min(1.0, final_risk + agreement_bonus)

                # 同时用加权平均值做参考上限
                components = []
                component_weights = []
                if global_fake_prob > 0:
                    components.append(global_fake_prob)
                    component_weights.append(w_global)
                if local_tamper_prob > 0:
                    components.append(local_tamper_prob)
                    component_weights.append(w_local)
                if metadata_risk > 0:
                    components.append(metadata_risk)
                    component_weights.append(0.15)
                if components:
                    total_w = sum(component_weights)
                    weighted_avg = sum(c * w / total_w for c, w in zip(components, component_weights))
                    final_risk = max(final_risk, weighted_avg * 0.7)

            final_risk = max(0.0, min(1.0, float(final_risk)))

            # ================== 5. 结果判定与理由梳理 ==================
            if global_fake_prob > thresh_global:
                reasons.append("全局UI布局异常")
            if pixel_anomaly > thresh_pixel_alert:
                reasons.append("存在局部边缘拼接/像素涂抹痕迹")
            if should_use_font_signal and font_anomaly > 0.55:
                reasons.append("局部字体风格异常")
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
                "reason": "；".join(reasons),
            }
            return json.dumps(output, ensure_ascii=False, indent=4, cls=NumpyEncoder)

        except Exception as e:
            logger.error(f"引擎推理引发未捕获异常: {e}", exc_info=True)
            error_output = {
                "result": "错误",
                "confidence": 0.0,
                "bbox": roi_bbox,
                "reason": f"引擎内部解析失败: {str(e)}",
            }
            return json.dumps(error_output, ensure_ascii=False, indent=4, cls=NumpyEncoder)


if __name__ == "__main__":
    import time

    logger.info("启动单图推理本地测试 (Inference API)")

    try:
        engine = InferenceEngineAPI(config_path="config.yaml")
        logger.info("引擎初始化成功")
    except Exception as e:
        logger.error(f"引擎初始化失败: {e}", exc_info=True)
        exit(1)

    test_image_path = "images/no (11).jpg"
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
