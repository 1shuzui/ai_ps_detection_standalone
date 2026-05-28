import os
import cv2
import numpy as np
import joblib
from io import BytesIO
from PIL import Image, ExifTags


class PixelLevelDetector:
    """像素级篡改检测器：ELA + 拉普拉斯边缘 + 生成器假图 + 噪声一致性 + 颜色一致性 + DCT 频域分析"""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.generator_bg_var_threshold = cfg.get("generator_bg_var_threshold", 0.05)
        self.generator_penalty = cfg.get("generator_penalty", 0.70)
        self.generator_enabled = cfg.get("generator_enabled", True)
        self.noise_consistency_weight = cfg.get("noise_consistency_weight", 0.15)
        self.color_consistency_weight = cfg.get("color_consistency_weight", 0.10)
        self.dct_analysis_enabled = cfg.get("dct_analysis_enabled", True)
        self.dct_weight = cfg.get("dct_weight", 0.12)

    def detect(self, cropped_img_np, quality=85, surrounding_np=None, neighbor_rois=None):
        if cropped_img_np is None or cropped_img_np.size == 0:
            return 0.0

        # 1. ELA 检测
        img_pil = Image.fromarray(cv2.cvtColor(cropped_img_np, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        img_pil.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        ela_img = np.abs(np.array(img_pil).astype(np.int16) - np.array(Image.open(buffer)).astype(np.int16))
        ela_gray = np.max(ela_img, axis=2)
        ela_mean = np.mean(ela_gray) / 255.0
        ela_std = np.std(ela_gray) / 128.0
        ela_score = (ela_mean * (1 + ela_std)) * 2.0

        # 2. 拉普拉斯高频突变检测
        gray = cv2.cvtColor(cropped_img_np, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        h, w = gray.shape
        edge_penalty = 0.0
        generator_penalty = 0.0
        noise_consistency_penalty = 0.0
        color_consistency_penalty = 0.0
        dct_penalty = 0.0

        if h > 20 and w > 20:
            # 2a. 生成器假图检测（可配置）
            if self.generator_enabled:
                mask = np.ones((h, w), dtype=bool)
                mask[5:-5, 5:-5] = False
                bg_pixels = gray[mask]
                bg_var = np.var(bg_pixels)
                if bg_var < self.generator_bg_var_threshold:
                    generator_penalty = self.generator_penalty

            # 2b. 传统边缘接缝检测
            core = laplacian[10:-10, 10:-10]
            core_var = np.var(core)
            total_var = np.var(laplacian)
            if core_var > 0:
                noise_diff_ratio = abs(total_var - core_var) / core_var
                edge_penalty = min(0.4, noise_diff_ratio * 0.3)

            # 2c. 噪声模式一致性检测（ROI vs 周围背景）
            if surrounding_np is not None and surrounding_np.size > 0:
                noise_consistency_penalty = self._check_noise_consistency(gray, surrounding_np)

            # 2d. 颜色一致性检测（相邻 ROI 之间）
            if neighbor_rois and len(neighbor_rois) > 0:
                color_consistency_penalty = self._check_color_consistency(cropped_img_np, neighbor_rois)

            # 2e. DCT 频域异常检测（AI 生成/修补图像）
            dct_penalty = 0.0
            if self.dct_analysis_enabled:
                dct_penalty = self._check_dct_anomaly(gray)

        score = ela_score + edge_penalty + generator_penalty + noise_consistency_penalty + color_consistency_penalty + dct_penalty
        return float(min(1.0, score))

    def _check_noise_consistency(self, roi_gray: np.ndarray, surrounding_np: np.ndarray) -> float:
        """检测 ROI 内部噪声模式与周围背景是否一致。不一致 -> 疑似粘贴拼接。"""
        try:
            sur_gray = cv2.cvtColor(surrounding_np, cv2.COLOR_BGR2GRAY) if surrounding_np.ndim == 3 else surrounding_np
            sur_gray = cv2.resize(sur_gray, (roi_gray.shape[1], roi_gray.shape[0]))

            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
            roi_noise = cv2.filter2D(roi_gray.astype(np.float32), -1, kernel)
            sur_noise = cv2.filter2D(sur_gray.astype(np.float32), -1, kernel)

            roi_std = float(np.std(roi_noise))
            sur_std = float(np.std(sur_noise))
            if roi_std < 1e-6 and sur_std < 1e-6:
                return 0.0

            ratio = min(roi_std, sur_std) / max(roi_std, sur_std, 1e-6)
            if ratio < 0.3:
                return self.noise_consistency_weight
            return 0.0
        except cv2.error:
            return 0.0

    def _check_color_consistency(self, roi_bgr: np.ndarray, neighbor_rois: list[np.ndarray]) -> float:
        """检测相邻 ROI 之间的颜色分布是否一致。不一致 -> 疑似不同来源拼接。"""
        try:
            roi_hist = self._color_histogram(roi_bgr)
            max_divergence = 0.0
            for neighbor in neighbor_rois:
                if neighbor is None or neighbor.size == 0:
                    continue
                neighbor_hist = self._color_histogram(neighbor)
                divergence = float(np.sum(np.abs(roi_hist - neighbor_hist)))
                max_divergence = max(max_divergence, divergence)

            if max_divergence > 0.5:
                return self.color_consistency_weight
            return 0.0
        except cv2.error:
            return 0.0

    @staticmethod
    def _color_histogram(bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 256, 0, 256])
        hist = hist / (hist.sum() + 1e-10)
        return hist.flatten()

    def _check_dct_anomaly(self, gray: np.ndarray) -> float:
        """8x8 分块 DCT 频域分析，检测 AI 生成/修补图像的高频异常。

        AI 生成或 inpainting 图像的 DCT 高频系数分布异常均匀或过低，
        与自然图像/真实拍摄有明显差异。
        """
        if not self.dct_analysis_enabled:
            return 0.0
        try:
            h, w = gray.shape
            if h < 16 or w < 16:
                return 0.0

            gray_f = gray.astype(np.float32)
            block_size = 8
            h_blocks = h // block_size
            w_blocks = w // block_size
            if h_blocks < 2 or w_blocks < 2:
                return 0.0

            high_freq_energies = []
            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = gray_f[i * block_size:(i + 1) * block_size,
                                   j * block_size:(j + 1) * block_size]
                    dct_block = cv2.dct(block)
                    zigzag = np.array([dct_block[k, l]
                                       for k in range(block_size)
                                       for l in range(block_size)
                                       if k + l > 4])
                    energy = float(np.std(zigzag))
                    high_freq_energies.append(energy)

            energies = np.array(high_freq_energies)
            mean_energy = float(np.mean(energies))
            std_energy = float(np.std(energies))

            # 高频能量过低 -> 生成式平滑
            if mean_energy < 0.8:
                return self.dct_weight
            # 高频能量分布异常均匀 -> 非自然拍摄
            if mean_energy > 0 and (std_energy / mean_energy) < 0.15:
                return self.dct_weight * 0.7
            return 0.0
        except cv2.error:
            return 0.0


class OriginalityChecker:
    """原图与 EXIF 校验器"""

    KNOWN_EDITING_SOFTWARE = ["photoshop", "picsart", "美图", "snapseed", "lightroom", "canva", "illustrator", "gimp"]

    def __init__(self, model_path=None):
        self.model = joblib.load(model_path) if model_path and os.path.exists(model_path) else None

    @staticmethod
    def extract_features(image_path):
        feats = {
            "has_exif": 0, "exif_count": 0, "time_diff": 0, "noise_std": 0,
            "noise_mean": 0, "noise_skew": 0, "size_per_pixel": 0, "color_entropy": 0,
        }
        hard_rule_tampered = False
        suspicious_software = ""

        if not os.path.exists(image_path):
            return None, False, ""

        try:
            img_pil = Image.open(image_path)
            exif = img_pil._getexif()
            if exif:
                feats["has_exif"] = 1
                feats["exif_count"] = len(exif)
                exif_dict = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
                if "EXIF DateTimeOriginal" in exif_dict and "EXIF DateTimeDigitized" in exif_dict:
                    feats["time_diff"] = 1
                software = str(exif_dict.get("Software", "")).lower()
                for bad in OriginalityChecker.KNOWN_EDITING_SOFTWARE:
                    if bad in software:
                        hard_rule_tampered = True
                        suspicious_software = software
                        break
        except Exception:
            pass

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
            noise = cv2.filter2D(img.astype(np.float32), -1, kernel)
            feats["noise_std"] = float(np.std(noise))
            feats["noise_mean"] = float(np.mean(np.abs(noise)))
            feats["noise_skew"] = float(
                np.mean((noise - feats["noise_mean"]) ** 3) / (feats["noise_std"] ** 3 + 1e-10)
            )
            h, w = img.shape
            feats["size_per_pixel"] = float(os.path.getsize(image_path) / (h * w) if (h * w) > 0 else 0)

        img_color = cv2.imread(image_path)
        if img_color is not None:
            hist = cv2.calcHist([img_color], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            feats["color_entropy"] = float(-np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0)

        return feats, hard_rule_tampered, suspicious_software

    def predict(self, image_path):
        feats, hard_rule, software = self.extract_features(image_path)
        if feats is None:
            return 0.0, False, ""
        if hard_rule:
            return 1.0, True, f"EXIF检测到修图软件: {software}"
        if self.model:
            prob = float(self.model.predict_proba(np.array([list(feats.values())]))[0][1])
            return prob, False, ""
        return 0.5, False, ""

    @staticmethod
    def compute_metadata_risk(feats: dict | None) -> tuple[float, list[str]]:
        """基于 EXIF/元数据特征计算风险分和理由，供推理管线直接使用。"""
        if feats is None:
            return 0.0, []
        risk = 0.0
        reasons = []
        if feats.get("has_exif", 0) == 0 and feats.get("size_per_pixel", 0.0) >= 0.18 and feats.get("color_entropy", 99.0) <= 2.2:
            risk = max(risk, 0.45)
            reasons.append("缺少EXIF且图像结构异常(疑似生成图)")
        if feats.get("color_entropy", 99.0) <= 1.8:
            risk = max(risk, 0.55)
        if feats.get("size_per_pixel", 0.0) >= 0.3:
            risk = max(risk, 0.50)
            reasons.append("文件体积/像素比异常(疑似工具导出)")
        return risk, reasons
