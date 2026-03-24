"""
AI 图像鉴伪服务核心网关 (Enterprise Production-Ready Edition)

新增生产级特性：
1. 异步高并发推理锁 (防止 GPU OOM)
2. 自动化垃圾回收守护进程 (防止磁盘打满)
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
import re
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import easyocr
import uvicorn
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, ImageDraw, ImageFont

from inference_api import InferenceEngineAPI

# ==========================================
# 0. 全局配置与基础设施初始化
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STORAGE_DIR = "/tmp/tamper_detector_storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# 生产级安全配置
MAX_CONCURRENT_AI_TASKS = 1  # 严格限制同时访问 GPU 的任务数，1为绝对安全防 OOM，视显存可调为 2 或 3
GC_MAX_AGE_HOURS = 24        # 文件及任务保留的最长时间 (小时)
GC_INTERVAL_SECONDS = 3600   # 垃圾回收轮询间隔 (1小时)

# ==========================================
# 1. 数据契约与防腐层 (Data Contracts / ACL)
# ==========================================
class TaskStatusEnum(str, Enum):
    UPLOADED = "UPLOADED"
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

class BBoxDTO(BaseModel):
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(gt=0)
    y2: int = Field(gt=0)
    model_config = ConfigDict(strict=True)

class TaskRecordDTO(BaseModel):
    task_id: str
    status: TaskStatusEnum
    created_at: str
    image_path: Optional[str] = None
    bbox: Optional[BBoxDTO] = None
    result: Optional[Dict[str, Any]] = None
    multi_results: Optional[List[Dict[str, Any]]] = None
    error_msg: Optional[str] = None

class PaginatedHistoryDTO(BaseModel):
    total: int
    page: int
    size: int
    items: List[TaskRecordDTO]

# ==========================================
# 2. 基础设施层 (Infrastructure & Registries)
# ==========================================
class AbstractTaskRegistry(ABC):
    @abstractmethod
    async def create_task(self, task_id: str, image_path: str) -> None: pass
    @abstractmethod
    async def update_task(self, task_id: str, **kwargs) -> None: pass
    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[TaskRecordDTO]: pass
    @abstractmethod
    async def delete_task(self, task_id: str) -> bool: pass
    @abstractmethod
    async def list_tasks(self, page: int, size: int) -> PaginatedHistoryDTO: pass

class MemoryTaskRegistry(AbstractTaskRegistry):
    def __init__(self):
        self._store: Dict[str, TaskRecordDTO] = {}

    async def create_task(self, task_id: str, image_path: str) -> None:
        self._store[task_id] = TaskRecordDTO(
            task_id=task_id, status=TaskStatusEnum.UPLOADED, 
            created_at=datetime.now().isoformat(), image_path=image_path
        )

    async def update_task(self, task_id: str, **kwargs) -> None:
        if task_id in self._store:
            task = self._store[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)

    async def get_task(self, task_id: str) -> Optional[TaskRecordDTO]:
        return self._store.get(task_id)

    async def delete_task(self, task_id: str) -> bool:
        if task_id in self._store:
            img_path = self._store[task_id].image_path
            if img_path and os.path.exists(img_path): os.remove(img_path)
            del self._store[task_id]
            return True
        return False

    async def list_tasks(self, page: int, size: int) -> PaginatedHistoryDTO:
        tasks = list(self._store.values())
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        start, end = (page - 1) * size, page * size
        return PaginatedHistoryDTO(total=len(tasks), page=page, size=size, items=tasks[start:end])

# --- 自动化垃圾回收守护进程 ---
async def cleanup_daemon(registry: AbstractTaskRegistry, storage_dir: str):
    logger.info(f"Garbage Collection Daemon started. (Interval: {GC_INTERVAL_SECONDS}s, Max Age: {GC_MAX_AGE_HOURS}h)")
    while True:
        try:
            await asyncio.sleep(GC_INTERVAL_SECONDS)
            logger.info("Running scheduled garbage collection...")
            now = datetime.now()
            
            if isinstance(registry, MemoryTaskRegistry):
                tasks_to_delete = []
                for task_id, task in registry._store.items():
                    try:
                        created_time = datetime.fromisoformat(task.created_at)
                        if (now - created_time) > timedelta(hours=GC_MAX_AGE_HOURS):
                            tasks_to_delete.append(task_id)
                    except Exception as e:
                        logger.warning(f"Error parsing date for task {task_id}: {e}")
                        
                for t_id in tasks_to_delete:
                    await registry.delete_task(t_id) # 会一并删除原图
                    # 额外清理可能存在的可视化结果图
                    vis_path = os.path.join(storage_dir, f"vis_{t_id}.jpg")
                    if os.path.exists(vis_path):
                        os.remove(vis_path)
                        
                if tasks_to_delete:
                    logger.info(f"Garbage Collection cleared {len(tasks_to_delete)} expired tasks and their files.")
        except asyncio.CancelledError:
            logger.info("Garbage Collection Daemon stopped.")
            break
        except Exception as e:
            logger.error(f"Error in Garbage Collection Daemon: {e}")

class EngineContainer:
    instance: Optional[InferenceEngineAPI] = None
    registry: Optional[AbstractTaskRegistry] = None
    ocr_reader: Optional[Any] = None
    ai_semaphore: Optional[asyncio.Semaphore] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing AI Infrastructure (Engine + OCR + Registry + Semaphore)...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading EasyOCR on device: {device}")
        EngineContainer.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=(device == 'cuda'))
        EngineContainer.instance = InferenceEngineAPI(config_path="config.yaml")
        EngineContainer.registry = MemoryTaskRegistry()
        EngineContainer.ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AI_TASKS)
        logger.info("AI Infrastructure loaded successfully.")
        
        # 挂载后台垃圾回收任务
        cleanup_task = asyncio.create_task(cleanup_daemon(EngineContainer.registry, STORAGE_DIR))
        
    except Exception as e:
        logger.error("Failed to load AI Infrastructure.", exc_info=True)
        raise RuntimeError("Initialization failed.") from e
    
    yield  # 服务运行期
    
    # 优雅停机卸载资源
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
        
    EngineContainer.instance = None
    EngineContainer.registry = None
    EngineContainer.ocr_reader = None
    EngineContainer.ai_semaphore = None

def get_engine() -> InferenceEngineAPI:
    if not EngineContainer.instance: raise HTTPException(503, "Engine unavailable")
    return EngineContainer.instance

def get_registry() -> AbstractTaskRegistry:
    if not EngineContainer.registry: raise HTTPException(503, "Registry unavailable")
    return EngineContainer.registry

def get_ocr_reader() -> Any:
    if not EngineContainer.ocr_reader: raise HTTPException(503, "OCR unavailable")
    return EngineContainer.ocr_reader

def get_ai_semaphore() -> asyncio.Semaphore:
    if not EngineContainer.ai_semaphore: raise HTTPException(503, "Semaphore unavailable")
    return EngineContainer.ai_semaphore

# ==========================================
# 3. 领域服务层 (Domain Service)
# ==========================================

class DetectionService:
    """【原版保留】V1 同步鉴伪核心业务服务"""
    @staticmethod
    async def process_detection(file: UploadFile, bbox_list: List[int], engine: InferenceEngineAPI, semaphore: asyncio.Semaphore) -> Dict:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
                
            # 引入并发锁保护
            async with semaphore:
                result_str = await run_in_threadpool(engine.predict, tmp_path, bbox_list)
                
            result_dict = json.loads(result_str)
            if result_dict.get("result") == "错误":
                raise ValueError(result_dict.get("reason", "Unknown engine internal error."))
            return result_dict
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


class DetectionDomainServiceV3:
    """【新增】V3 异步领域服务 (支持全图多框独立检测与并发锁保护)"""
    def __init__(self, engine: InferenceEngineAPI, registry: AbstractTaskRegistry, ocr_reader: Any, semaphore: asyncio.Semaphore):
        self.engine = engine
        self.registry = registry
        self.ocr_reader = ocr_reader
        self.semaphore = semaphore

    def _easyocr_auto_detect(self, image_path: str) -> List[BBoxDTO]:
        logger.info(f"Running EasyOCR multi-bbox detection for {image_path}")
        img_cv2 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_cv2 is None: return []

        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 3) 
        
        ocr_results = self.ocr_reader.readtext(blurred, adjust_contrast=0.5, mag_ratio=2.0, text_threshold=0.25)

        bboxes = []
        for bbox, text, conf in ocr_results:
            text_clean = text.replace(" ", "")
            total_len = len(text_clean)
            if total_len == 0: continue

            digits_count = len(re.findall(r'\d', text_clean))
            digit_ratio = digits_count / total_len

            if digits_count < 3 or (total_len > 18 and digit_ratio < 0.5):
                continue

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            bboxes.append(BBoxDTO(x1=int(min(xs)), y1=int(min(ys)), x2=int(max(xs)), y2=int(max(ys))))
        
        return bboxes

    async def execute_async(self, task_id: str, image_path: str, bbox: Optional[BBoxDTO] = None) -> None:
        task = await self.registry.get_task(task_id)
        if not task or task.status == TaskStatusEnum.CANCELED: return
        await self.registry.update_task(task_id, status=TaskStatusEnum.PROCESSING)
        
        try:
            # 场景 A：用户指定单框检测
            if bbox:
                bbox_list = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
                
                # 引入并发锁保护单次推理
                async with self.semaphore:
                    res_str = await run_in_threadpool(self.engine.predict, image_path, bbox_list)
                    
                res_dict = json.loads(res_str)
                if res_dict.get("result") == "错误": raise ValueError(res_dict.get("reason"))
                res_dict["original_bbox"] = bbox_list
                await self.registry.update_task(task_id, status=TaskStatusEnum.COMPLETED, result=res_dict)
            
            # 场景 B：全图自动扫描多框验证
            else:
                # OCR同样受并发锁保护，防止多个大图同时加载进显存
                async with self.semaphore:
                    bboxes = await run_in_threadpool(self._easyocr_auto_detect, image_path)
                    
                if not bboxes:
                    empty_res = {"result": "正常", "confidence": 0.0, "reason": "未发现关键数值或单号区域"}
                    await self.registry.update_task(task_id, status=TaskStatusEnum.COMPLETED, result=empty_res)
                    return

                all_results = []
                for b in bboxes:
                    try:
                        b_list = [b.x1, b.y1, b.x2, b.y2]
                        # 对循环切片推理施加严格的并发控制，多请求排队等待
                        async with self.semaphore:
                            res_str = await run_in_threadpool(self.engine.predict, image_path, b_list)
                            
                        res_dict = json.loads(res_str)
                        if res_dict.get("result") != "错误":
                            res_dict["original_bbox"] = b_list
                            all_results.append(res_dict)
                    except Exception as e:
                        logger.warning(f"Task {task_id}: Single bbox {b} prediction failed: {e}")
                
                await self.registry.update_task(task_id, status=TaskStatusEnum.COMPLETED, multi_results=all_results)

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            await self.registry.update_task(task_id, status=TaskStatusEnum.FAILED, error_msg=str(e))

    async def generate_visualization(self, task_id: str) -> str:
        task = await self.registry.get_task(task_id)
        if not task or task.status != TaskStatusEnum.COMPLETED:
            raise ValueError("Task not completed.")
        
        vis_path = os.path.join(STORAGE_DIR, f"vis_{task_id}.jpg")
        if os.path.exists(vis_path): return vis_path

        def draw_bboxes():
            img_cv2 = cv2.imdecode(np.fromfile(task.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.truetype("simhei.ttf", 22)
            except IOError:
                logger.warning("simhei.ttf not found. Using default font.")
                font = ImageFont.load_default()

            results_to_draw = task.multi_results if task.multi_results else []
            if task.result and not task.multi_results:
                results_to_draw.append(task.result)

            for res in results_to_draw:
                original_b = res.get("original_bbox")
                if not original_b: original_b = res.get("bbox", [0,0,10,10])
                x1, y1, x2, y2 = original_b[0], original_b[1], original_b[2], original_b[3]

                real_status = res.get("result", "正常")
                confidence = res.get("confidence", 0.0)

                if real_status == "篡改":
                    color, text_color = (255, 0, 0), (255, 255, 255)
                    label = f"篡改 | 风险:{confidence:.1%}"
                elif real_status == "可疑":
                    color, text_color = (255, 165, 0), (0, 0, 0)
                    label = f"可疑 | 风险:{confidence:.1%}"
                else:
                    color, text_color = (0, 255, 0), (0, 0, 0)
                    label = f"正常 | 风险:{confidence:.1%}"

                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                label_bg_y1 = max(y1 - text_height - 6, 0)
                
                draw.rectangle([(x1, label_bg_y1), (min(x1 + text_width + 6, img_pil.width), max(y1, text_height + 6))], fill=color)
                draw.text((x1 + 3, label_bg_y1 + 3), label, font=font, fill=text_color)
                
            img_cv2_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cv2.imencode('.jpg', img_cv2_result)[1].tofile(vis_path)

        await run_in_threadpool(draw_bboxes)
        return vis_path


# ==========================================
# 4. 接口路由层 (API Controllers)
# ==========================================
app = FastAPI(title="AI Image Tampering Detection API", version="3.2.0-Prod", lifespan=lifespan)

# ---- V1 兼容路由 ----
@app.post("/api/v1/image-detection/detect")
async def detect_tampering_endpoint(
    file: UploadFile = File(...), 
    bbox: str = Form(...), 
    engine: InferenceEngineAPI = Depends(get_engine),
    semaphore: asyncio.Semaphore = Depends(get_ai_semaphore)
):
    try:
        clean_bbox = bbox.strip().strip("'").strip('"').strip()
        bbox_parsed = json.loads(clean_bbox) if clean_bbox.startswith('[') else [int(x.strip()) for x in clean_bbox.split(',')]
        if len(bbox_parsed) != 4: raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bbox format.")
    try:
        res = await DetectionService.process_detection(file, [int(x) for x in bbox_parsed], engine, semaphore)
        return {"status": "success", "data": res}
    except ValueError as ve:
        return JSONResponse(status_code=422, content={"status": "error", "message": str(ve)})

# ---- V3 新版路由 ----
api_router_v3 = APIRouter(prefix="/api/v3", tags=["V3 Pipeline"])

@api_router_v3.post("/detect", summary="提交检测任务")
async def submit_detection(
    background_tasks: BackgroundTasks,
    task_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    bbox: Optional[str] = Form(None),
    engine: InferenceEngineAPI = Depends(get_engine),
    registry: AbstractTaskRegistry = Depends(get_registry),
    ocr_reader: Any = Depends(get_ocr_reader),
    semaphore: asyncio.Semaphore = Depends(get_ai_semaphore)
):
    if file:
        task_id = str(uuid.uuid4())
        file_path = os.path.join(STORAGE_DIR, f"{task_id}.jpg")
        with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        await registry.create_task(task_id, file_path)
    elif not task_id:
        raise HTTPException(status_code=400, detail="Must provide task_id or file.")

    task = await registry.get_task(task_id)
    if not task: raise HTTPException(404, "Task not found.")

    bbox_dto = None
    if bbox:
        arr = json.loads(bbox) if bbox.startswith('[') else [int(x.strip()) for x in bbox.split(',')]
        bbox_dto = BBoxDTO(x1=arr[0], y1=arr[1], x2=arr[2], y2=arr[3])

    await registry.update_task(task_id, status=TaskStatusEnum.PENDING)
    service = DetectionDomainServiceV3(engine, registry, ocr_reader, semaphore)
    background_tasks.add_task(service.execute_async, task_id, task.image_path, bbox_dto)
    return {"status": "pending", "task_id": task_id}

@api_router_v3.get("/result/{task_id}", response_model=TaskRecordDTO)
async def get_result(task_id: str, registry: AbstractTaskRegistry = Depends(get_registry)):
    task = await registry.get_task(task_id)
    if not task: raise HTTPException(404, "Task not found")
    return task

@api_router_v3.get("/result/{task_id}/visualization")
async def get_visualization(
    task_id: str, 
    engine: InferenceEngineAPI = Depends(get_engine), 
    registry: AbstractTaskRegistry = Depends(get_registry), 
    ocr_reader: Any = Depends(get_ocr_reader),
    semaphore: asyncio.Semaphore = Depends(get_ai_semaphore)
):
    service = DetectionDomainServiceV3(engine, registry, ocr_reader, semaphore)
    try:
        vis_path = await service.generate_visualization(task_id)
        return FileResponse(vis_path, media_type="image/jpeg")
    except ValueError as e:
        raise HTTPException(400, str(e))

@api_router_v3.delete("/task/{task_id}")
async def cancel_task(task_id: str, registry: AbstractTaskRegistry = Depends(get_registry)):
    task = await registry.get_task(task_id)
    if not task: raise HTTPException(404, "Task not found")
    if task.status in [TaskStatusEnum.PENDING, TaskStatusEnum.UPLOADED]:
        await registry.update_task(task_id, status=TaskStatusEnum.CANCELED)
    else:
        await registry.delete_task(task_id)
    return {"status": "success"}

app.include_router(api_router_v3)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7000, workers=1)