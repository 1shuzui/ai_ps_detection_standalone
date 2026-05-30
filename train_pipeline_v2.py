"""
训练管线 — 支持反馈数据增强、模型版本化、可视化。
使用 /api/v3/train 端点触发，训练前有风险提示。
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import joblib
import numpy as np
import re
import xgboost as xgb
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from PIL import Image

from core.augmentations import build_global_augmentations, build_roi_augmentations
from core.detectors import OriginalityChecker
from core.extractors import FeatureExtractor, FontFeatureLibrary

logger = logging.getLogger(__name__)


class TrainPipeline:
    """端到端训练管线 — 全局模型 + 字体库 + 可视化。"""

    def __init__(self, config_path: str = "config.yaml", ocr_reader: Any = None):
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = (Path(__file__).resolve().parent / config_file).resolve()
        with open(config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.base_dir = config_file.parent

        train_cfg = self.config.get("training", {})
        self.output_dir = Path(self._resolve(train_cfg.get("output_dir", "models/trained")))
        self.viz_enabled = train_cfg.get("visualization_enabled", True)
        self.viz_dir = Path(self._resolve(train_cfg.get("visualization_dir", "models/trained/viz")))
        self.backup_previous = train_cfg.get("backup_previous", True)

        self.ocr_reader = ocr_reader
        self.extractor: Optional[FeatureExtractor] = None
        self.font_lib: Optional[FontFeatureLibrary] = None

    def _resolve(self, path_str: str) -> str:
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str((self.base_dir / p).resolve())

    def run(
        self,
        feedback_dir: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """执行完整训练流程，返回训练摘要。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.viz_enabled:
            self.viz_dir.mkdir(parents=True, exist_ok=True)

        # 备份旧模型
        if self.backup_previous:
            self._backup_models()

        # 初始化提取器
        if self.extractor is None:
            self.extractor = FeatureExtractor(reader=self.ocr_reader)

        feedback_wrong = self._load_feedback_wrong(feedback_dir)
        original_images = self._load_original_dataset()

        all_font_feats = []
        all_font_labels = []
        global_X = []
        global_y = []

        total_samples = len(original_images) + len(feedback_wrong)
        processed = 0

        # 处理原始数据集
        for img_path, label in original_images:
            try:
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                # 全局特征 + 增强
                augs = build_global_augmentations(img, key=os.path.basename(img_path))
                for aug_name, aug_img in augs:
                    feat = self.extractor.extract_global_feature(aug_img)
                    global_X.append(feat)
                    global_y.append(label)

                # 字体特征：仅正样本（label=0，无篡改）
                if label == 0:
                    self._collect_font_features(img, img_path, all_font_feats, all_font_labels)

                processed += 1
                if progress_callback:
                    progress_callback(processed, total_samples, f"处理原始数据: {os.path.basename(img_path)}")
            except Exception:
                logger.warning("处理图像失败: %s", img_path, exc_info=True)

        # 处理反馈的错误样本（作为篡改样本加入训练，label=1）
        for fb_img_path, metadata in feedback_wrong:
            try:
                img = cv2.imdecode(np.fromfile(fb_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                augs = build_global_augmentations(img, key=os.path.basename(fb_img_path))
                for aug_name, aug_img in augs:
                    feat = self.extractor.extract_global_feature(aug_img)
                    global_X.append(feat)
                    global_y.append(1)

                processed += 1
                if progress_callback:
                    progress_callback(processed, total_samples, f"处理反馈数据: {os.path.basename(fb_img_path)}")
            except Exception:
                logger.warning("处理反馈图像失败: %s", fb_img_path, exc_info=True)

        if not global_X:
            return {"status": "failed", "reason": "没有可用于训练的样本"}

        # 训练字体库
        font_lib_path = str(self.output_dir / "font_lib")
        if all_font_feats:
            new_font_lib = FontFeatureLibrary()
            new_font_lib.add(all_font_feats, all_font_labels)
            new_font_lib.save(font_lib_path)
            logger.info("字体库已保存: %s (共 %d 条)", font_lib_path, len(all_font_labels))

        # 训练全局模型（含 Platt 校准 + 早停）
        X = np.array(global_X)
        y = np.array(global_y)

        # 划分验证集用于早停
        if len(X) >= 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = X, None, y, None

        base_model = xgb.XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
        )

        if X_val is not None and len(X_val) > 0:
            base_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            base_model.fit(X_train, y_train)

        # Platt scaling 校准：将 XGBoost 原始概率映射到校准概率
        model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
        model.fit(X_train, y_train)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = str(self.output_dir / f"global_layout_model_{timestamp}.pkl")
        joblib.dump(model, model_path)

        # 同时保存最新版本
        latest_path = str(self.output_dir / "global_layout_model.pkl")
        joblib.dump(model, latest_path)

        # 可视化
        viz_paths = []
        if self.viz_enabled:
            viz_paths = self._generate_visualizations(model, X, y, timestamp)

        summary = {
            "status": "completed",
            "timestamp": timestamp,
            "total_samples": len(global_y),
            "positive_samples": int(sum(global_y)),
            "negative_samples": int(len(global_y) - sum(global_y)),
            "font_library_size": len(all_font_labels),
            "model_path": model_path,
            "font_lib_path": font_lib_path,
            "visualizations": viz_paths,
            "train_accuracy": float(model.score(X, y)),
        }

        # 保存训练摘要
        with open(self.output_dir / f"train_summary_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 写入模型版本注册表
        self._update_registry(
            timestamp=timestamp,
            model_path=model_path,
            font_lib_path=font_lib_path,
            train_accuracy=float(model.score(X, y)),
            total_samples=len(global_y),
        )

        logger.info("训练完成: %s", json.dumps(summary, ensure_ascii=False))
        return summary

    def _update_registry(self, timestamp: str, model_path: str, font_lib_path: str,
                         train_accuracy: float, total_samples: int) -> None:
        """更新模型版本注册表 registry.json。"""
        registry_path = Path(self._resolve(
            self.config.get("training", {}).get("registry_path", "models/registry.json")
        ))
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": timestamp,
            "model_path": model_path,
            "font_lib_path": font_lib_path,
            "train_accuracy": train_accuracy,
            "total_samples": total_samples,
        }

        registry = {"versions": []}
        if registry_path.exists():
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
            except (json.JSONDecodeError, OSError):
                registry = {"versions": []}

        registry.setdefault("versions", []).append(entry)
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        logger.info("已更新模型注册表: %s (共 %d 个版本)", registry_path, len(registry["versions"]))

    def _backup_models(self):
        """备份旧模型到带时间戳的子目录。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.output_dir / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ["global_layout_model.pkl", "font_lib.index", "font_lib_meta.pkl"]:
            src = self.output_dir / pattern
            if src.exists():
                shutil.copy2(str(src), str(backup_dir / pattern))

    def _load_feedback_wrong(self, feedback_dir: Optional[str]) -> list:
        """加载 feedback/wrong 目录中的错误样本。"""
        samples = []
        if not feedback_dir:
            feedback_dir = str(Path(self._resolve(self.config.get("feedback", {}).get("storage_dir", "feedback"))) / "wrong")
        fb_path = Path(feedback_dir)
        if not fb_path.exists():
            return samples
        for folder in fb_path.iterdir():
            meta_file = folder / "metadata.json"
            orig = folder / "original.jpg"
            if meta_file.exists() and orig.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                samples.append((str(orig), metadata))
        return samples

    def _load_original_dataset(self) -> list:
        """加载原始 images/ 数据集并分配标签。"""
        dataset_cfg = self.config.get("dataset", {})
        img_dir = Path(self._resolve(dataset_cfg.get("image_dir", "images")))
        samples = []
        if img_dir.exists():
            for path in sorted(img_dir.iterdir()):
                if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                label = 0 if "no" in path.name.lower() else 1
                samples.append((str(path), label))
        return samples

    def _collect_font_features(self, img, img_path, all_feats, all_labels):
        """从正样本图像中采集字体特征。"""
        json_dir = Path(self._resolve(self.config.get("dataset", {}).get("json_dir", "locate_json")))
        json_path = json_dir / (Path(img_path).stem + ".json")
        if not json_path.exists():
            return

        with open(json_path, "r", encoding="utf-8") as f:
            regions_data = json.load(f)

        roi_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seen = set()
        for region in regions_data.get("key_regions", []):
            if region.get("type") != "amount":
                continue
            text = region.get("text", "")
            if len(re.findall(r"\d", str(text))) < 3:
                continue
            bbox = region.get("bbox", [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            key = (x1, y1, x2, y2, str(text))
            if key in seen:
                continue
            seen.add(key)
            char_img = roi_rgb[y1:y2, x1:x2]
            if char_img.size == 0:
                continue
            augs = build_roi_augmentations(char_img, key=f"{img_path}_{x1}_{y1}")
            for aug_name, aug_img in augs:
                tensor = self.extractor.transform_local(aug_img)
                if self.extractor.device.type == "cuda":
                    tensor = tensor.cpu()
                feat = self.extractor.feature_extractor(tensor.unsqueeze(0).to(self.extractor.device))
                feat_np = feat.view(-1).cpu().numpy()
                all_feats.append(feat_np)
                all_labels.append(str(text)[:20])

    def _generate_visualizations(self, model, X, y, timestamp) -> list:
        """生成训练可视化图片并保存。"""
        import re

        paths = []
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # 1. 特征重要性图
            fig, ax = plt.subplots(figsize=(10, 6))
            importance = model.feature_importances_
            indices = np.argsort(importance)[-30:]
            ax.barh(range(len(indices)), importance[indices])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([f"dim_{i}" for i in indices], fontsize=7)
            ax.set_xlabel("Importance")
            ax.set_title(f"XGBoost Feature Importance (top 30)\n{timestamp}")
            fig.tight_layout()
            imp_path = str(self.viz_dir / f"feature_importance_{timestamp}.png")
            fig.savefig(imp_path, dpi=120)
            plt.close(fig)
            paths.append(imp_path)

            # 2. 训练分数分布
            probs = model.predict_proba(X)[:, 1]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.hist(probs[y == 0], bins=30, alpha=0.6, label="Real (label=0)", color="green")
            ax1.hist(probs[y == 1], bins=30, alpha=0.6, label="Fake (label=1)", color="red")
            ax1.set_xlabel("Predicted Probability")
            ax1.set_ylabel("Count")
            ax1.set_title("Score Distribution by Class")
            ax1.legend()

            y_pred = model.predict(X)
            from sklearn.metrics import confusion_matrix as cm
            cmat = cm(y, y_pred)
            im = ax2.imshow(cmat, cmap="Blues")
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(["Real", "Fake"])
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(["Real", "Fake"])
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    ax2.text(j, i, str(cmat[i, j]), ha="center", va="center", fontsize=14)
            ax2.set_title("Confusion Matrix")
            plt.colorbar(im, ax=ax2)

            fig.tight_layout()
            dist_path = str(self.viz_dir / f"training_analysis_{timestamp}.png")
            fig.savefig(dist_path, dpi=120)
            plt.close(fig)
            paths.append(dist_path)

            # 3. 学习曲线采样图
            fig, ax = plt.subplots(figsize=(8, 5))
            train_sizes = np.linspace(0.1, 1.0, 10)
            scores = []
            for frac in train_sizes:
                n = max(10, int(len(X) * frac))
                idx = np.random.choice(len(X), n, replace=False)
                sub_model = xgb.XGBClassifier(max_depth=4, n_estimators=100, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
                sub_model.fit(X[idx], y[idx])
                scores.append(sub_model.score(X, y))
            ax.plot(train_sizes * 100, scores, "b-o")
            ax.set_xlabel("Training Data %")
            ax.set_ylabel("Accuracy")
            ax.set_title("Learning Curve (approx)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            curve_path = str(self.viz_dir / f"learning_curve_{timestamp}.png")
            fig.savefig(curve_path, dpi=120)
            plt.close(fig)
            paths.append(curve_path)

        except Exception:
            logger.warning("生成可视化失败", exc_info=True)

        return paths
