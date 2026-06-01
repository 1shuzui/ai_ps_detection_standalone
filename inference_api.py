import cv2
import json
import math
import time
import yaml
import numpy as np
import logging
import os
import joblib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.exceptions import DetectionError, RecoverableError
from core.extractors import FeatureExtractor, FontFeatureLibrary, TamperAnalyzer
from core.detectors import PixelLevelDetector, OriginalityChecker
from core.utils import NumpyEncoder, safe_read_image

logger = logging.getLogger(__name__)


class InferenceEngineAPI:
    def __init__(self, config_path="config.yaml", shared_ocr_reader: Optional[Any] = None):
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = (Path(__file__).resolve().parent / config_file).resolve()

        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.base_dir = config_file.parent

        preprocess_cfg = self.config.get("preprocessing", {})
        self.extractor = FeatureExtractor(
            reader=shared_ocr_reader,
            preserve_aspect_ratio=preprocess_cfg.get("preserve_aspect_ratio", True),
        )
        self.font_lib = FontFeatureLibrary()
        font_lib_path = self._resolve_path(self.config['paths']['font_lib_path'])
        self.font_lib.load(font_lib_path)

        xgb_path = self.config.get('paths', {}).get('xgb_model_path', "models/global_layout_model.pkl")
        self.global_model = joblib.load(self._resolve_path(xgb_path))

        pixel_cfg = self.config.get('pixel_detector', {})
        self.pixel_detector = PixelLevelDetector(config=pixel_cfg)

        self.originality_checker = OriginalityChecker()
        self._origin_enabled = self.config.get('originality', {}).get('enabled', True)

        self._font_lib_path = font_lib_path
        self._xgb_path = self._resolve_path(xgb_path)

        registry_path = self.config.get("training", {}).get("registry_path", "models/registry.json")
        self._registry_path = self._resolve_path(registry_path)

        calib_cfg = self.config.get("thresholds", {})
        self._calibration_temp = float(calib_cfg.get("calibration_temperature", 1.0))

        self._metrics: dict = {
            "total_predictions": 0,
            "tampered_count": 0,
            "suspicious_count": 0,
            "normal_count": 0,
            "error_count": 0,
            "total_inference_time_ms": 0.0,
            "inference_times_ms": [],
        }

    def list_model_versions(self) -> dict:
        """返回模型版本注册表中所有版本。"""
        registry_path = Path(self._registry_path)
        if not registry_path.exists():
            return {"versions": [], "current_model": str(self._xgb_path)}
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"versions": [], "current_model": str(self._xgb_path)}
        registry["current_model"] = str(self._xgb_path)
        return registry

    def reload_models(self, version: Optional[str] = None) -> dict:
        """热重载 FAISS 字体库和 XGBoost 模型，无需重启服务。

        指定 version 时从注册表中查找对应版本路径进行加载。
        Python 属性赋值是原子的，读取端无锁安全。
        """
        result = {"font_lib": "unchanged", "global_model": "unchanged"}

        xgb_path = self._xgb_path
        font_lib_path = self._font_lib_path

        if version:
            versions_info = self.list_model_versions()
            for entry in versions_info.get("versions", []):
                if entry.get("timestamp") == version:
                    xgb_path = entry.get("model_path", xgb_path)
                    font_lib_path = entry.get("font_lib_path", font_lib_path)
                    result["version"] = version
                    logger.info("切换到模型版本: %s", version)
                    break
            else:
                logger.warning("未找到版本 %s，使用当前活跃模型", version)
                result["version"] = "current"

        new_font_lib = FontFeatureLibrary()
        if new_font_lib.load(font_lib_path):
            self.font_lib = new_font_lib
            result["font_lib"] = "reloaded"
        else:
            logger.warning("字体库重载失败，保留当前库")

        try:
            new_model = joblib.load(xgb_path)
            self.global_model = new_model
            result["global_model"] = "reloaded"
        except Exception:
            logger.warning("全局模型重载失败，保留当前模型", exc_info=True)

        logger.info("模型热重载完成: %s", result)
        return result

    def get_metrics(self) -> dict:
        """返回累计推理指标快照。"""
        times = self._metrics.get("inference_times_ms", [])
        sorted_times = sorted(times) if times else []
        n = len(sorted_times)
        p50 = sorted_times[n // 2] if n > 0 else 0.0
        p99 = sorted_times[min(n - 1, int(n * 0.99))] if n > 0 else 0.0

        return {
            "total_predictions": self._metrics["total_predictions"],
            "tampered_count": self._metrics["tampered_count"],
            "suspicious_count": self._metrics["suspicious_count"],
            "normal_count": self._metrics["normal_count"],
            "error_count": self._metrics["error_count"],
            "inference_p50_ms": round(p50, 2),
            "inference_p99_ms": round(p99, 2),
            "avg_inference_ms": round(
                self._metrics["total_inference_time_ms"] / max(1, self._metrics["total_predictions"]), 2
            ),
            "font_lib_size": self.font_lib.index.ntotal,
            "font_lib_ready": self.font_lib.is_ready,
        }

    def _resolve_path(self, path_str: str) -> str:
        path = Path(path_str)
        if path.is_absolute():
            return str(path)
        return str((self.base_dir / path).resolve())

    @staticmethod
    def _calibrate_proba(raw_proba: float, temperature: float) -> float:
        """二次校准：温度缩放防止 XGBoost 极端置信度。temperature > 1 压低极端值。"""
        p = max(1e-3, min(1.0 - 1e-3, raw_proba))
        logit = math.log(p / (1.0 - p))
        return 1.0 / (1.0 + math.exp(-logit / temperature))

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

    @staticmethod
    def _has_hard_metadata_evidence(
        metadata_risk: float,
        metadata_reasons: List[str],
        threshold: float = 0.50,
    ) -> bool:
        if metadata_risk < threshold:
            return False
        hard_markers = (
            "EXIF检测到已知修图软件",
            "文件体积/像素比异常",
            "缺少EXIF且图像结构异常",
        )
        return any(
            any(marker in reason for marker in hard_markers)
            for reason in (metadata_reasons or [])
        )

    def predict(
        self,
        full_image_path: str,
        roi_bbox: List[int],
        bbox_format: str = "auto",
        precomputed_global_feat: Optional[np.ndarray] = None,
    ) -> str:
        try:
            _t0 = time.time()
            reasons = []
            result_status = "正常"

            img = safe_read_image(full_image_path)
            if img is None:
                self._metrics["error_count"] += 1
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
            metadata_reasons = []
            if self._origin_enabled:
                orig_feats, hard_rule, _ = self.originality_checker.extract_features(full_image_path)
                if hard_rule:
                    metadata_reasons.append("EXIF检测到已知修图软件")
                    reasons.extend(metadata_reasons)
                    metadata_risk = 0.55
                elif orig_feats:
                    m_risk, m_reasons = OriginalityChecker.compute_metadata_risk(orig_feats)
                    metadata_risk = m_risk
                    metadata_reasons = list(m_reasons)
                    reasons.extend(metadata_reasons)

            # ================== 1. 全局特征分析 ==================
            if precomputed_global_feat is not None:
                global_feat = precomputed_global_feat
            else:
                global_feat = self.extractor.extract_global_feature(img)
            global_fake_prob_raw = float(self.global_model.predict_proba(np.array([global_feat]))[0][1])
            global_fake_prob = self._calibrate_proba(global_fake_prob_raw, self._calibration_temp)

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

            # 字体库冷启动降级：库为空时跳过字体信号，像素权重自动提升
            if not self.font_lib.is_ready:
                should_use_font_signal = False

            # 批量字体相似度查询
            font_sims = self.font_lib.search_similarity_batch(feats) if (feats and self.font_lib.is_ready) else []
            font_sim = np.mean(font_sims) if font_sims else 0.5
            font_anomaly = max(0.0, 1.0 - font_sim)

            # 字体异常分级：仅当 ROI 包含核心文本且匹配度足够时才计入
            if font_sim < 0.3:
                effective_font_anomaly = 0.0       # 字体不在库中
            elif font_sim < 0.5:
                effective_font_anomaly = font_anomaly * 0.3  # 中匹配缩放
            else:
                effective_font_anomaly = font_anomaly         # 高匹配全量

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
                    + effective_font_anomaly * weights.get('core_font', 0.4)
                    + geo_penalty
                )
                if text_profile["digit_count"] >= 8 and effective_font_anomaly > 0.75:
                    local_tamper_prob = max(local_tamper_prob, thresh_low + 0.02)
            else:
                local_tamper_prob = (pixel_anomaly * weights.get('non_core_pixel', 0.8)) + geo_penalty
                if pixel_anomaly < thresh_exempt and geo_penalty == 0:
                    local_tamper_prob = 0.0

            local_tamper_prob = max(0.0, min(1.0, float(local_tamper_prob)))

            # ================== 4. 融合策略：两层 AI 篡改检测 ==================
            # 第一层：全图 AI 生成特征（全局模型）
            global_ai_score = global_fake_prob

            # 第二层：ROI 局部异常（像素 + 字体 + 几何）
            local_anomaly = max(local_tamper_prob, float(geo_penalty))

            # 融合：全局 AI 信号 + 局部异常交叉验证
            if global_ai_score > thresh_global and local_anomaly > 0.50:
                # 全局和局部都异常 → AI 局部篡改，高置信度
                final_risk = (global_ai_score + local_anomaly) / 2.0
            elif global_ai_score > thresh_global:
                # 仅全局异常 → EXIF + 元数据辅助判定可信度
                _has_exif = 0
                if self._origin_enabled:
                    _of, _, _ = self.originality_checker.extract_features(full_image_path)
                    _has_exif = _of.get("has_exif", 0) if _of else 0

                # 元数据强证据时提升全局信号（有EXIF+低色彩熵=AI生成特征）
                _effective_global = global_ai_score
                # 元数据强证据时提升全局信号
                if _has_exif and metadata_risk >= 0.50 and 0.68 <= global_ai_score < 0.85:
                    _effective_global = min(0.95, global_ai_score + 0.25)

                # EXIF 分级乘数：全局越确信，乘数越高
                if _has_exif:
                    if _effective_global >= 0.75:
                        g_mult = 0.73
                    else:
                        g_mult = 0.60
                elif _effective_global > 0.90:
                    g_mult = 0.70
                else:
                    g_mult = 0.50
                final_risk = _effective_global * g_mult + local_anomaly * (1.0 - g_mult) * 0.5
            else:
                # 全局正常 → 局部 + 元数据兜底
                if local_anomaly > 0.80:
                    final_risk = local_anomaly * 0.90
                else:
                    final_risk = local_anomaly * 0.70
                # 元数据强异常兜底（排除纯色彩均匀，白底文档常见）
                if metadata_risk >= 0.50 and any(
                    r for r in reasons if "EXIF" in r or "体积" in r or "结构" in r
                ):
                    final_risk = max(final_risk, 0.52)

            metadata_hard_tamper = self._has_hard_metadata_evidence(
                metadata_risk,
                metadata_reasons,
                float(thresh.get("metadata_hard_evidence", 0.50)),
            )
            if metadata_hard_tamper:
                final_risk = max(final_risk, float(thresh_high) + 0.02)

            final_risk = max(0.0, min(1.0, float(final_risk)))

            # ================== 5. 结果判定与理由梳理 ==================
            if global_fake_prob > thresh_global:
                reasons.append("全局UI布局异常")
            if pixel_anomaly > thresh_pixel_alert:
                reasons.append("存在局部边缘拼接/像素涂抹痕迹")
            if should_use_font_signal and font_anomaly > 0.65:
                reasons.append("局部字体风格异常")
            if geo_penalty > 0:
                reasons.extend(geo_reasons)

            if metadata_hard_tamper:
                result_status = "篡改"
            elif final_risk > thresh_high:
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

            elapsed_ms = (time.time() - _t0) * 1000.0
            self._metrics["total_predictions"] += 1
            self._metrics["total_inference_time_ms"] += elapsed_ms
            self._metrics["inference_times_ms"].append(elapsed_ms)
            if result_status == "篡改":
                self._metrics["tampered_count"] += 1
            elif result_status == "可疑":
                self._metrics["suspicious_count"] += 1
            else:
                self._metrics["normal_count"] += 1

            return json.dumps(output, ensure_ascii=False, indent=4, cls=NumpyEncoder)

        except RecoverableError as e:
            self._metrics["error_count"] += 1
            logger.warning("可恢复的业务异常: %s", e)
            error_output = {
                "result": "错误",
                "confidence": 0.0,
                "bbox": roi_bbox,
                "reason": str(e),
            }
            return json.dumps(error_output, ensure_ascii=False, indent=4, cls=NumpyEncoder)
        except Exception as e:
            self._metrics["error_count"] += 1
            logger.error("引擎推理引发未捕获系统异常: %s", e, exc_info=True)
            error_output = {
                "result": "错误",
                "confidence": 0.0,
                "bbox": roi_bbox,
                "reason": "引擎内部解析失败，请联系运维排查",
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
