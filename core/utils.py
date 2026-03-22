import json
import numpy as np
from datetime import datetime
import cv2
from PIL import Image, ImageDraw, ImageFont

# 兼容性修复：安全的NumPy类型检查
NP_BOOL_TYPES = (np.bool_,) if hasattr(np, 'bool_') else (bool,)
NP_INT_TYPES = (np.integer, np.int64, np.int32, np.int16, np.int8)
NP_FLOAT_TYPES = (np.floating, np.float64, np.float32, np.float16)

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

def put_chinese_text(img_rgb, text, position, text_color=(255, 255, 255), font_size=20):
    """利用 PIL 在图片上画中文（替代 cv2.putText 避免中文乱码）"""
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=text_color)
    return np.array(img_pil)

import os
import cv2
import numpy as np

def safe_read_image(image_path: str) -> np.ndarray:
    """
    安全读取图片，完美支持包含中文、空格、特殊字符的路径。
    替代容易暴雷的 cv2.imread
    """
    if not os.path.exists(image_path):
        return None
    # 使用 numpy 从文件流读取，再交给 cv2 解码，绕过底层 C++ 的路径编码限制
    return cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)