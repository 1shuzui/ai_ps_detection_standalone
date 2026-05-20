import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 兼容性修复：安全的NumPy类型检查
NP_BOOL_TYPES = (np.bool_,) if hasattr(np, 'bool_') else (bool,)
NP_INT_TYPES = (np.integer, np.int64, np.int32, np.int16, np.int8)
NP_FLOAT_TYPES = (np.floating, np.float64, np.float32, np.float16)
FONT_CANDIDATES = (
    "msyh.ttf",
    "MSYH.TTF",
    "SimHei.ttf",
    "simhei.ttf",
    "NotoSansCJK-Regular.ttc",
    "NotoSansCJK.ttc",
    "SourceHanSansCN-Regular.otf",
)
SYSTEM_FONT_DIRS = (
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path.home() / ".local" / "share" / "fonts",
)

class NumpyEncoder(json.JSONEncoder):
    """安全处理NumPy类型，专为 FastAPI/JSON 响应设计"""
    def default(self, obj):
        if isinstance(obj, NP_BOOL_TYPES):
            return bool(obj)
        elif isinstance(obj, NP_INT_TYPES):
            return int(obj)
        elif isinstance(obj, NP_FLOAT_TYPES):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, (datetime,)):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        return super(NumpyEncoder, self).default(obj)


@lru_cache(maxsize=1)
def resolve_chinese_font_path() -> Optional[str]:
    """在仓库和常见系统字体目录中查找可用中文字体。"""
    search_roots = [
        Path.cwd(),
        Path(__file__).resolve().parent.parent,
    ]

    for root in search_roots:
        for candidate in FONT_CANDIDATES:
            font_path = root / candidate
            if font_path.exists():
                return str(font_path)

    for font_dir in SYSTEM_FONT_DIRS:
        if not font_dir.exists():
            continue
        for candidate in FONT_CANDIDATES:
            matches = list(font_dir.rglob(candidate))
            if matches:
                return str(matches[0])

    return None


def load_chinese_font(size: int = 20) -> ImageFont.ImageFont:
    font_path = resolve_chinese_font_path()
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass

    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue

    return ImageFont.load_default()


def put_chinese_text(img_rgb, text, position, text_color=(255, 255, 255), font_size=20):
    """利用 PIL 在图片上画中文（替代 cv2.putText 避免中文乱码）"""
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = load_chinese_font(font_size)
    draw.text(position, text, font=font, fill=text_color)
    return np.array(img_pil)

def safe_read_image(image_path: str) -> np.ndarray:
    """
    安全读取图片，完美支持包含中文、空格、特殊字符的路径。
    替代容易暴雷的 cv2.imread
    """
    if not os.path.exists(image_path):
        return None
    # 使用 numpy 从文件流读取，再交给 cv2 解码，绕过底层 C++ 的路径编码限制
    return cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
