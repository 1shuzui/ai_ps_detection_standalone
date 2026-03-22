import os
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import easyocr
import numpy as np
import faiss
import pickle
import logging
import re

logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 预加载 OCR，只实例化一次
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=(self.device.type == 'cuda'))

        # 加载 ResNet 用于提取特征
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.feature_extractor.eval()

        self.transform_local = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform_global = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract_global_feature(self, img_np):
        if img_np is None or img_np.size == 0:
            return np.zeros(512, dtype=np.float32)

        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        tensor = self.transform_global(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.feature_extractor(tensor)
        return feat.view(-1).cpu().numpy()

    def extract_from_roi(self, roi_rgb):
        if roi_rgb is None or roi_rgb.size == 0:
            return [], []

        ocr_results = self.reader.readtext(roi_rgb)

        valid_stats = []
        tensor_list = []

        # 记录哪些索引是真正需要提取字体特征的（纯数字/金额）
        feature_indices = []

        for idx, (bbox, text, conf) in enumerate(ocr_results):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            h, w = roi_rgb.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # 【核心优化 1：精准打击】
            # 判断这个框里是不是包含数字（剔除纯汉字如“净重”、“吨”）
            has_digit = bool(re.search(r'\d', text))

            # 保存所有的排版信息（用于检测对齐）
            stat_info = {
                'text': text,
                'bbox': [x1, y1, x2, y2],
                'conf': conf,
                'is_core_number': has_digit
            }
            valid_stats.append(stat_info)

            # 只有包含数字的框，才截图喂给 ResNet 提特征
            if has_digit:
                char_img = roi_rgb[y1:y2, x1:x2]
                if char_img.size > 0:
                    tensor = self.transform_local(char_img)
                    tensor_list.append(tensor)
                    feature_indices.append(idx)

        # Batch 批量推理
        feats_list = []
        if tensor_list:
            batch_tensor = torch.stack(tensor_list).to(self.device)
            with torch.no_grad():
                batch_feats = self.feature_extractor(batch_tensor)

            batch_feats = batch_feats.view(batch_feats.size(0), -1).cpu().numpy()
            feats_list = [feat for feat in batch_feats]

        return feats_list, valid_stats


class TamperAnalyzer:
    @staticmethod
    def check_internal_consistency(valid_stats):
        """
        【核心优化 2：财务级排版审核】
        """
        reasons = []
        score_penalty = 0.0

        # 只抽取出核心数字的统计信息进行排版对比
        num_stats = [s for s in valid_stats if s.get('is_core_number', False)]

        if len(num_stats) < 2:
            return reasons, score_penalty

        heights = [s['bbox'][3] - s['bbox'][1] for s in num_stats]
        y_coords = [s['bbox'][1] for s in num_stats]

        # 1. 高度一致性校验（防拼凑大数字）
        h_mean = np.mean(heights)
        h_variance = np.var(heights)
        if h_mean > 0 and (h_variance / h_mean) > 3.0:  # 容忍一定误差，超过 3.0 报警
            reasons.append("数值区域高度突变(疑似大小字拼接)")
            score_penalty += 0.20

        # 2. 基线（Y轴）一致性校验（防上下错位拼接）
        y_variance = np.var(y_coords)
        if y_variance > 15:  # 同一行的数字，Y坐标方差不应超过 15 像素
            reasons.append("数值区域基线严重不齐(疑似错位拼接)")
            score_penalty += 0.25

        return reasons, float(min(0.5, score_penalty))  # 排版惩罚最高不超过 0.5


class FontFeatureLibrary:
    def __init__(self, dim=512):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.char_labels = []

    def add(self, feats, texts):
        if len(feats) == 0: return
        self.index.add(np.array(feats).astype('float32'))
        self.char_labels.extend(texts)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}_meta.pkl", 'wb') as f:
            pickle.dump({'char_labels': self.char_labels}, f)

    def load(self, path):
        if not os.path.exists(f"{path}.index"): return False
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}_meta.pkl", 'rb') as f:
            meta = pickle.load(f)
            self.char_labels = meta['char_labels']
        return True

    def search_similarity(self, query_feat):
        if self.index.ntotal == 0: return 0.5
        D, I = self.index.search(np.array([query_feat]).astype('float32'), 1)
        dist = D[0][0]
        sim = np.exp(-dist / 100.0)
        return float(np.clip(sim, 0.0, 1.0))