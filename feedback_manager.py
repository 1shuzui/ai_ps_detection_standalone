"""
人工标注反馈管理模块
- 支持 correct / wrong / suspicious 三级标注
- wrong: 保存原图 + 框选区域 + 元数据
- suspicious: 存入待确认目录，确认后移回 correct/wrong
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml


class FeedbackManager:
    """管理用户对检测结果的标注反馈。"""

    def __init__(self, config_path: str = "config.yaml"):
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = (Path(__file__).resolve().parent / config_file).resolve()
        with open(config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        fb_cfg = self.config.get("feedback", {})
        self.base_dir = Path(self._resolve_path(fb_cfg.get("storage_dir", "feedback")))
        self.correct_dir = self.base_dir / "correct"
        self.wrong_dir = self.base_dir / "wrong"
        self.suspicious_dir = self.base_dir / "suspicious"

        for d in [self.correct_dir, self.wrong_dir, self.suspicious_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path_str: str) -> str:
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str((Path(__file__).resolve().parent / p).resolve())

    def save_judgment(
        self,
        task_id: str,
        judgment: str,  # "correct" | "wrong" | "suspicious"
        image_path: str,
        bbox: Optional[List[int]] = None,
        result: Optional[Dict[str, Any]] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        """保存用户判断。wrong 时额外保存裁剪区域图。"""
        entry_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{task_id}_{entry_id}"

        if judgment == "wrong":
            target_dir = self.wrong_dir / folder_name
        elif judgment == "suspicious":
            target_dir = self.suspicious_dir / folder_name
        else:
            target_dir = self.correct_dir / folder_name

        target_dir.mkdir(parents=True, exist_ok=True)

        # 复制原图
        src_img = Path(image_path)
        dst_img = target_dir / f"original{src_img.suffix}"
        if src_img.exists():
            shutil.copy2(str(src_img), str(dst_img))

        # wrong: 额外保存裁剪区域图
        cropped_path = None
        if judgment in ("wrong", "suspicious") and bbox and src_img.exists():
            try:
                img = cv2.imdecode(np.fromfile(str(src_img), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    x1, y1, x2, y2 = bbox
                    x1, x2 = sorted([max(0, x1), min(img.shape[1], x2)])
                    y1, y2 = sorted([max(0, y1), min(img.shape[0], y2)])
                    roi = img[y1:y2, x1:x2]
                    cropped_path = str(target_dir / "roi.jpg")
                    cv2.imencode(".jpg", roi)[1].tofile(cropped_path)
            except Exception:
                pass

        # 保存元数据
        metadata = {
            "task_id": task_id,
            "judgment": judgment,
            "timestamp": timestamp,
            "bbox": bbox,
            "engine_result": result,
            "user_note": note,
            "entry_id": entry_id,
            "original_image": str(dst_img),
            "cropped_roi": cropped_path,
        }
        with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return metadata

    def confirm_suspicious(self, entry_folder: str, final_judgment: str) -> Optional[Dict[str, Any]]:
        """将 suspicious 条目确认后移入 correct 或 wrong 目录。"""
        src = self.suspicious_dir / entry_folder
        if not src.exists():
            return None

        meta_file = src / "metadata.json"
        if not meta_file.exists():
            return None

        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if final_judgment == "wrong":
            dst = self.wrong_dir / entry_folder
        else:
            dst = self.correct_dir / entry_folder

        if dst.exists():
            shutil.rmtree(str(dst))
        shutil.move(str(src), str(dst))

        metadata["judgment"] = final_judgment
        metadata["confirmed_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(dst / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return metadata

    def list_entries(self, judgment_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出反馈条目。"""
        entries = []
        search_dirs = []
        if judgment_filter:
            mapping = {"correct": self.correct_dir, "wrong": self.wrong_dir, "suspicious": self.suspicious_dir}
            d = mapping.get(judgment_filter)
            if d:
                search_dirs.append(d)
        else:
            search_dirs = [self.correct_dir, self.wrong_dir, self.suspicious_dir]

        for d in search_dirs:
            if not d.exists():
                continue
            for folder in sorted(d.iterdir(), reverse=True):
                meta_file = folder / "metadata.json"
                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        entries.append(json.load(f))

        return entries
