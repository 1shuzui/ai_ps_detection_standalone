"""
AI 图像鉴伪服务主入口 (Enterprise Edition)

提供高并发、高鲁棒性的 FastAPI Web 容器，负责 HTTP 协议解析、
模型生命周期管理、服务依赖注入及请求分发。
"""

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from inference_api import InferenceEngineAPI

# ==========================================
# 1. 基础设施层 (Infrastructure)
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

class EngineContainer:
    """引擎单例容器，用于跨生命周期管理资源。"""
    instance: InferenceEngineAPI = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器。
    负责在服务启动时挂载 AI 引擎，并在服务终止时释放资源。
    """
    logger.info("Initializing AI Inference Engine...")
    try:
        # 实际生产中，配置路径应通过环境变量注入
        EngineContainer.instance = InferenceEngineAPI(config_path="config.yaml")
        logger.info("AI Inference Engine loaded successfully.")
    except Exception as e:
        logger.error("Failed to load AI Inference Engine.", exc_info=True)
        raise RuntimeError("Engine initialization failed.") from e
    
    yield  # 服务运行期
    
    logger.info("Shutting down AI Inference Engine and releasing resources...")
    EngineContainer.instance = None


# ==========================================
# 2. 依赖注入层 (Dependency Injection)
# ==========================================

def get_inference_engine() -> InferenceEngineAPI:
    """
    引擎依赖注入提供者。
    可测试性保障：单元测试时可通过 app.dependency_overrides 覆写此方法。
    """
    if EngineContainer.instance is None:
        logger.error("Inference engine accessed before initialization.")
        raise HTTPException(status_code=503, detail="AI Service is currently unavailable.")
    return EngineContainer.instance


# ==========================================
# 3. 领域服务层 (Domain Service)
# ==========================================

class DetectionService:
    """
    鉴伪核心业务服务。
    隔离底层基础组件与上层 HTTP 协议，负责核心业务逻辑编排。
    """

    @staticmethod
    async def process_detection(
        file: UploadFile, 
        bbox_list: List[int], 
        engine: InferenceEngineAPI
    ) -> Dict[str, Any]:
        """
        处理单张图片的鉴伪请求。

        Args:
            file: 包含图像数据的上传文件对象。
            bbox_list: 目标检测区域坐标 [x1, y1, x2, y2]。
            engine: 鉴伪推理引擎实例。

        Returns:
            引擎推理结果字典。

        Raises:
            ValueError: 当引擎返回错误结果时抛出。
            IOError: 文件读写异常时抛出。
        """
        tmp_path = None
        try:
            # 采用安全临时文件管理
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            # 派发至线程池执行计算密集型推理，避免阻塞 asyncio 事件循环
            result_str = await run_in_threadpool(engine.predict, tmp_path, bbox_list)
            result_dict = json.loads(result_str)

            if result_dict.get("result") == "错误":
                raise ValueError(result_dict.get("reason", "Unknown engine internal error."))

            return result_dict

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as e:
                    logger.warning("Failed to clean up temporary file: %s. Error: %s", tmp_path, e)


# ==========================================
# 4. 接口路由层 (API Controllers)
# ==========================================

app = FastAPI(
    title="AI Image Tampering Detection API",
    description="Enterprise-grade unified API for image tampering verification.",
    version="2.0.0",
    lifespan=lifespan
)

@app.post("/api/v1/image-detection/detect")
async def detect_tampering_endpoint(
    file: UploadFile = File(..., description="Image file to detect (jpg/png)."),
    bbox: str = Form(..., description="JSON array or comma-separated string, e.g., '[10,20,100,50]' or '10,20,100,50'."),
    engine: InferenceEngineAPI = Depends(get_inference_engine)
):
    """
    图像篡改检测端点。
    接收前端图像流与坐标，调用 AI 模型评估该区域的真实性。
    """
    # 1. 表现层数据校验 (DTO Validation) - 【企业级高容错解析】
    try:
        # 预处理：剥离各种 HTTP 客户端可能私自添加的干扰字符（单双引号、空格）
        clean_bbox = bbox.strip().strip("'").strip('"').strip()
        
        # 策略 A：标准 JSON 数组解析 "[150, 200, 180, 45]"
        if clean_bbox.startswith('[') and clean_bbox.endswith(']'):
            bbox_parsed = json.loads(clean_bbox)
        # 策略 B：逗号分隔的纯净降级解析 "150, 200, 180, 45"
        else:
            bbox_parsed = [int(x.strip()) for x in clean_bbox.split(',')]
            
        if not isinstance(bbox_parsed, list) or len(bbox_parsed) != 4:
            raise ValueError("Parsed bounding box does not contain exactly 4 coordinates.")
            
        bbox_list = [int(x) for x in bbox_parsed]
        
    except Exception as e:
        logger.warning("Bbox parsing rejected. Raw input: '%s', Error: %s", bbox, str(e))
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid bbox parameter. Received: '{bbox}'. Please use '[x, y, w, h]' or 'x, y, w, h'."
        )

    # 2. 委派给业务服务层处理
    try:
        logger.info("Processing detection request for file: %s, bbox: %s", file.filename, bbox_list)
        result_data = await DetectionService.process_detection(file, bbox_list, engine)
        
        return {
            "status": "success",
            "message": "Detection completed successfully.",
            "data": result_data
        }

    except ValueError as ve:
        logger.warning("Engine processing error for file %s: %s", file.filename, ve)
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "message": str(ve),
                "data": None
            }
        )
    except Exception as e:
        logger.error("Unexpected error during detection endpoint execution.", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during detection processing.")


# ==========================================
# 5. 启动入口 (Application Runner)
# ==========================================

if __name__ == "__main__":
    # 配置启动参数
    uvicorn_config = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": 7000,
        "reload": False,
        "workers": 1,  # AI模型通常占用大量显存，若非使用 Triton 等专门 Serving，避免多 worker 导致 OOM
        "access_log": True
    }
    logger.info("Starting Uvicorn server on %s:%s", uvicorn_config["host"], uvicorn_config["port"])
    uvicorn.run(**uvicorn_config)