import os
import cv2
import numpy as np
import joblib
from PIL import Image, ExifTags


class PixelLevelDetector:
    """终极版：捕捉拼接、高频突变以及 '生成器假图(零噪点)' 的检测器"""

    def detect(self, cropped_img_np, quality=85):
        if cropped_img_np is None or cropped_img_np.size == 0:
            return 0.0

        # 1. 基础 ELA 检测 (抓取传统 PS 拼接)
        from io import BytesIO
        img_pil = Image.fromarray(cv2.cvtColor(cropped_img_np, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        img_pil.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        ela_img = np.abs(np.array(img_pil).astype(np.int16) - np.array(Image.open(buffer)).astype(np.int16))
        ela_gray = np.max(ela_img, axis=2)

        ela_mean = np.mean(ela_gray) / 255.0
        ela_std = np.std(ela_gray) / 128.0
        ela_score = (ela_mean * (1 + ela_std)) * 2.0

        # 2. 拉普拉斯高频突变检测 (抓边缘生硬的贴图)
        gray = cv2.cvtColor(cropped_img_np, cv2.COLOR_BGR2GRAY)

        # 【新增：物理屏摄摩尔纹抗性装甲】
        # 利用 3x3 高斯核熔断屏幕像素点带来的高频周期性噪声
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        h, w = gray.shape
        edge_penalty = 0.0
        generator_penalty = 0.0

        if h > 20 and w > 20:
            # 取图片外围一圈的背景区域
            mask = np.ones((h, w), dtype=bool)
            mask[5:-5, 5:-5] = False
            bg_pixels = gray[mask]

            # 【核心杀招】：造假生成器的背景通常是绝对的纯色 (方差接近 0)
            # 真实截图经过微信等压缩，背景方差必然大于 0.1
            bg_var = np.var(bg_pixels)
            if bg_var < 0.05:
                # 如果背景平滑到极其不自然的地步，赋予极高的生成器假图惩罚分！
                generator_penalty = 0.70

                # 传统边缘接缝检测
            core = laplacian[10:-10, 10:-10]
            core_var = np.var(core)
            total_var = np.var(laplacian)
            if core_var > 0:
                noise_diff_ratio = abs(total_var - core_var) / core_var
                edge_penalty = min(0.4, noise_diff_ratio * 0.3)

        # 最终像素得分 = ELA得分 + 边缘贴图惩罚 + 生成器纯色惩罚
        score = ela_score + edge_penalty + generator_penalty
        return float(min(1.0, score))

class OriginalityChecker:
    """原图与 EXIF 校验器"""
    def __init__(self, model_path=None):
        self.model = joblib.load(model_path) if model_path and os.path.exists(model_path) else None

    @staticmethod
    def extract_features(image_path):
        feats = {'has_exif': 0, 'exif_count': 0, 'time_diff': 0, 'noise_std': 0,
                 'noise_mean': 0, 'noise_skew': 0, 'size_per_pixel': 0, 'color_entropy': 0}
        hard_rule_tampered = False
        suspicious_software = ''

        if not os.path.exists(image_path): return None, False, ""

        try:
            img_pil = Image.open(image_path)
            exif = img_pil._getexif()
            if exif:
                feats['has_exif'] = 1
                feats['exif_count'] = len(exif)
                exif_dict = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
                if 'EXIF DateTimeOriginal' in exif_dict and 'EXIF DateTimeDigitized' in exif_dict:
                    feats['time_diff'] = 1
                software = str(exif_dict.get('Software', '')).lower()
                bad_softwares = ['photoshop', 'picsart', '美图', 'snapseed', 'lightroom']
                for bad in bad_softwares:
                    if bad in software:
                        hard_rule_tampered = True
                        suspicious_software = software
                        break
        except:
            pass

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise = cv2.filter2D(img.astype(np.float32), -1, kernel)
            feats['noise_std'] = float(np.std(noise))
            feats['noise_mean'] = float(np.mean(np.abs(noise)))
            feats['noise_skew'] = float(np.mean((noise - feats['noise_mean']) ** 3) / (feats['noise_std'] ** 3 + 1e-10))
            h, w = img.shape
            feats['size_per_pixel'] = float(os.path.getsize(image_path) / (h * w) if (h * w) > 0 else 0)

        img_color = cv2.imread(image_path)
        if img_color is not None:
            hist = cv2.calcHist([img_color], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            feats['color_entropy'] = float(-np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0)

        return feats, hard_rule_tampered, suspicious_software

    def predict(self, image_path):
        feats, hard_rule, software = self.extract_features(image_path)
        if feats is None: return 0.0, False, ""
        if hard_rule: return 0.0, True, f"EXIF检测到修图软件: {software}"
        if self.model:
            prob = self.model.predict_proba(np.array([list(feats.values())]))[0][1]
            return float(prob), False, ""
        return 0.5, False, ""