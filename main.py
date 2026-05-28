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
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import easyocr
import uvicorn
import yaml
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, ImageDraw, ImageFont

from core.logging_config import configure_logging
from core.ocr_utils import build_amount_candidates, detect_certificate_document_override, tokenize_ocr_results
from core.utils import load_chinese_font
from inference_api import InferenceEngineAPI

# ==========================================
# 0. 全局配置与基础设施初始化
# ==========================================
_config_path = Path(__file__).resolve().parent / "config.yaml"
with open(_config_path, "r", encoding="utf-8") as _f:
    _app_config = yaml.safe_load(_f)

_startup_time = configure_logging(_app_config)
logger = logging.getLogger(__name__)

_server_cfg = _app_config.get("server", {})
STORAGE_DIR = str(Path(_server_cfg.get("storage_dir", "data/storage")).resolve())
os.makedirs(STORAGE_DIR, exist_ok=True)

MAX_CONCURRENT_AI_TASKS = _server_cfg.get("max_concurrent_tasks", 1)
GC_MAX_AGE_HOURS = _server_cfg.get("gc_max_age_hours", 24)
GC_INTERVAL_SECONDS = _server_cfg.get("gc_interval_seconds", 3600)

_ocr_cfg = _app_config.get("ocr", {})
OCR_ADJUST_CONTRAST = _ocr_cfg.get("adjust_contrast", 0.5)
OCR_MAG_RATIO = _ocr_cfg.get("mag_ratio", 2.0)
OCR_TEXT_THRESHOLD = _ocr_cfg.get("text_threshold", 0.25)

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
        ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=(device == 'cuda'))
        EngineContainer.ocr_reader = ocr_reader
        EngineContainer.instance = InferenceEngineAPI(config_path="config.yaml", shared_ocr_reader=ocr_reader)
        EngineContainer.registry = MemoryTaskRegistry()
        EngineContainer.ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AI_TASKS)
        logger.info("AI Infrastructure loaded successfully (shared OCR reader).")

        cleanup_task = asyncio.create_task(cleanup_daemon(EngineContainer.registry, STORAGE_DIR))

    except Exception as e:
        logger.error("Failed to load AI Infrastructure.", exc_info=True)
        raise RuntimeError("Initialization failed.") from e

    yield

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
    """V3 异步领域服务 (支持全图多框独立检测、并发锁保护、OCR 结果复用)"""
    def __init__(self, engine: InferenceEngineAPI, registry: AbstractTaskRegistry, ocr_reader: Any, semaphore: asyncio.Semaphore):
        self.engine = engine
        self.registry = registry
        self.ocr_reader = ocr_reader
        self.semaphore = semaphore
        self._cached_img_cv2 = None
        self._cached_tokens = None
        self._cached_candidates = None
        self._cached_global_feat = None

    @staticmethod
    def _bbox_iou(a: BBoxDTO, b: BBoxDTO) -> float:
        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union_area = max(area_a + area_b - inter_area, 1)
        return inter_area / union_area

    def _deduplicate_bboxes(self, bboxes: List[BBoxDTO], iou_threshold: float = 0.85) -> List[BBoxDTO]:
        deduped: List[BBoxDTO] = []
        for bbox in sorted(bboxes, key=lambda b: ((b.x2 - b.x1) * (b.y2 - b.y1)), reverse=True):
            if any(self._bbox_iou(bbox, kept) >= iou_threshold for kept in deduped):
                continue
            deduped.append(bbox)
        return deduped

    def _run_ocr_once(self, image_path: str):
        """读取图片并执行 OCR + 全局特征提取，结果缓存供后续复用。"""
        if self._cached_tokens is not None:
            return
        img_cv2 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_cv2 is None:
            return
        self._cached_img_cv2 = img_cv2
        self._cached_global_feat = self.engine.extractor.extract_global_feature(img_cv2)
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 3)
        ocr_results = self.ocr_reader.readtext(
            blurred,
            adjust_contrast=OCR_ADJUST_CONTRAST,
            mag_ratio=OCR_MAG_RATIO,
            text_threshold=OCR_TEXT_THRESHOLD,
        )
        self._cached_tokens = tokenize_ocr_results(ocr_results)
        self._cached_candidates = build_amount_candidates(self._cached_tokens, img_cv2.shape)

    def _easyocr_auto_detect(self, image_path: str) -> List[BBoxDTO]:
        logger.info(f"Running EasyOCR multi-bbox detection for {image_path}")
        self._run_ocr_once(image_path)
        if not self._cached_candidates:
            return []
        bboxes = [
            BBoxDTO(x1=int(c.bbox[0]), y1=int(c.bbox[1]), x2=int(c.bbox[2]), y2=int(c.bbox[3]))
            for c in self._cached_candidates
        ]
        return self._deduplicate_bboxes(bboxes)

    def _document_rule_override(self, image_path: str) -> Optional[Dict[str, Any]]:
        if self._cached_img_cv2 is None:
            self._run_ocr_once(image_path)
        if self._cached_img_cv2 is None or not self._cached_tokens:
            return None

        override = detect_certificate_document_override(
            image_path=Path(image_path),
            image=self._cached_img_cv2,
            tokens=self._cached_tokens,
            candidates=self._cached_candidates or [],
            ocr_reader=self.ocr_reader,
        )
        if not override:
            return None

        bbox = [int(value) for value in override["bbox"].split(",")]
        return {
            "result": override["status"],
            "confidence": float(override["confidence"]),
            "reason": override["reason"],
            "bbox": bbox,
            "original_bbox": bbox,
        }

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
                    res_str = await run_in_threadpool(self.engine.predict, image_path, bbox_list, "xyxy")
                    
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
                    async with self.semaphore:
                        document_override = await run_in_threadpool(self._document_rule_override, image_path)

                    if document_override:
                        await self.registry.update_task(
                            task_id,
                            status=TaskStatusEnum.COMPLETED,
                            result=document_override,
                            multi_results=[document_override],
                        )
                    else:
                        empty_res = {"result": "正常", "confidence": 0.0, "reason": "未发现关键数值或单号区域"}
                        await self.registry.update_task(task_id, status=TaskStatusEnum.COMPLETED, result=empty_res)
                    return

                global_feat = self._cached_global_feat
                all_results = []
                for b in bboxes:
                    try:
                        b_list = [b.x1, b.y1, b.x2, b.y2]
                        # 对循环切片推理施加严格的并发控制，多请求排队等待
                        async with self.semaphore:
                            res_str = await run_in_threadpool(
                                self.engine.predict, image_path, b_list, "xyxy", global_feat
                            )
                            
                        res_dict = json.loads(res_str)
                        if res_dict.get("result") != "错误":
                            res_dict["original_bbox"] = b_list
                            all_results.append(res_dict)
                    except Exception as e:
                        logger.warning(f"Task {task_id}: Single bbox {b} prediction failed: {e}")

                async with self.semaphore:
                    document_override = await run_in_threadpool(self._document_rule_override, image_path)
                if document_override and not any(item.get("result") == "篡改" for item in all_results):
                    all_results.append(document_override)
                
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
            font = load_chinese_font(22)

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

# ---- 人工标注反馈系统 ----
class JudgmentRequest(BaseModel):
    task_id: str
    judgment: str = Field(..., pattern="^(correct|wrong|suspicious)$")
    bbox: Optional[List[int]] = None
    note: str = ""


@api_router_v3.post("/feedback/judge", summary="提交人工判断标注")
async def submit_judgment(
    req: JudgmentRequest,
    registry: AbstractTaskRegistry = Depends(get_registry),
):
    from feedback_manager import FeedbackManager

    task = await registry.get_task(req.task_id)
    if not task:
        raise HTTPException(404, "任务不存在")

    fb = FeedbackManager()
    result = task.result or {}
    bbox = req.bbox
    if bbox is None:
        bbox = result.get("original_bbox") or result.get("bbox")
    entry = fb.save_judgment(
        task_id=req.task_id,
        judgment=req.judgment,
        image_path=task.image_path,
        bbox=bbox,
        result=result,
        note=req.note,
    )
    logger.info("反馈已保存: task=%s judgment=%s entry=%s", req.task_id, req.judgment, entry.get("entry_id"))
    return {"status": "success", "entry": entry}


@api_router_v3.get("/feedback/list", summary="列出反馈记录")
async def list_feedback(judgment: Optional[str] = Query(None, pattern="^(correct|wrong|suspicious)$")):
    from feedback_manager import FeedbackManager

    fb = FeedbackManager()
    entries = fb.list_entries(judgment_filter=judgment)
    return {"total": len(entries), "items": entries}


@api_router_v3.post("/feedback/confirm", summary="确认疑似标注转向")
async def confirm_suspicious(folder_name: str = Form(...), judgment: str = Form(..., pattern="^(correct|wrong)$")):
    from feedback_manager import FeedbackManager

    fb = FeedbackManager()
    entry = fb.confirm_suspicious(folder_name, judgment)
    if not entry:
        raise HTTPException(404, "疑似条目不存在或已处理")
    return {"status": "success", "entry": entry}


# ---- 训练端点 (含风险提示) ----
class TrainResponse(BaseModel):
    status: str
    warning: str = "训练将使用反馈数据+原始数据集重新训练模型。这将覆盖当前模型（旧模型自动备份）。训练期间 GPU 资源占用高，可能影响正在进行的检测任务。"
    summary: Optional[Dict[str, Any]] = None


@api_router_v3.post("/train", summary="触发模型训练（含风险警告）")
async def trigger_training(
    confirm: bool = Form(False, description="必须设为 true 以确认风险"),
    engine: InferenceEngineAPI = Depends(get_engine),
    ocr_reader: Any = Depends(get_ocr_reader),
):
    if not confirm:
        return TrainResponse(
            status="aborted",
            warning="请仔细阅读风险提示，确认后将 confirm 设为 true 重新提交。",
        )

    from train_pipeline_v2 import TrainPipeline

    try:
        # training runs in thread pool to avoid blocking
        summary = await run_in_threadpool(
            lambda: TrainPipeline(ocr_reader=ocr_reader).run()
        )

        # 训练完成后自动将产出复制到活跃模型路径并热重载
        if summary.get("status") == "completed":
            import shutil
            trained_model = summary.get("model_path")
            trained_font_lib = summary.get("font_lib_path")
            if trained_model:
                active_model = engine._xgb_path
                shutil.copy2(trained_model, active_model)
                logger.info("已复制训练模型到活跃路径: %s", active_model)
            if trained_font_lib:
                import glob
                for src_pattern in [f"{trained_font_lib}.index", f"{trained_font_lib}_meta.pkl"]:
                    src = src_pattern
                    dst = f"{engine._font_lib_path}{src_pattern[len(trained_font_lib):]}"
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                logger.info("已复制字体库到活跃路径: %s", engine._font_lib_path)
            engine.reload_models()

        return TrainResponse(status="completed", summary=summary)
    except Exception as e:
        logger.exception("训练失败")
        return TrainResponse(status="failed", warning=f"训练异常: {str(e)}")


# ---- 训练可视化图片获取 ----
@api_router_v3.get("/train/viz/{filename}", summary="获取训练可视化图片")
async def get_train_visualization(filename: str):
    from train_pipeline_v2 import TrainPipeline

    pipeline = TrainPipeline()
    viz_file = pipeline.viz_dir / filename
    if not viz_file.exists():
        raise HTTPException(404, "可视化图片不存在")
    return FileResponse(str(viz_file), media_type="image/png")


@api_router_v3.get("/models", summary="列出所有模型版本")
async def list_models(engine: InferenceEngineAPI = Depends(get_engine)):
    return engine.list_model_versions()


@api_router_v3.post("/reload", summary="热重载模型（无需重启服务）")
async def reload_models(
    version: Optional[str] = Form(None, description="指定版本时间戳，不传则加载当前活跃版本"),
    engine: InferenceEngineAPI = Depends(get_engine),
):
    result = engine.reload_models(version=version)
    return {"status": "success", "detail": result}


@api_router_v3.get("/metrics", summary="推理指标监控（Prometheus 格式）")
async def get_metrics(engine: InferenceEngineAPI = Depends(get_engine)):
    m = engine.get_metrics()
    lines = [
        "# HELP tamper_detection_predictions_total Total number of predictions.",
        "# TYPE tamper_detection_predictions_total counter",
        f"tamper_detection_predictions_total {m['total_predictions']}",
        f"tamper_detection_tampered_total {m['tampered_count']}",
        f"tamper_detection_suspicious_total {m['suspicious_count']}",
        f"tamper_detection_normal_total {m['normal_count']}",
        f"tamper_detection_errors_total {m['error_count']}",
        "# HELP tamper_detection_inference_ms Inference latency in milliseconds.",
        "# TYPE tamper_detection_inference_ms gauge",
        f"tamper_detection_inference_p50_ms {m['inference_p50_ms']}",
        f"tamper_detection_inference_p99_ms {m['inference_p99_ms']}",
        f"tamper_detection_inference_avg_ms {m['avg_inference_ms']}",
        "# HELP tamper_detection_font_lib_info Font library status.",
        "# TYPE tamper_detection_font_lib_info gauge",
        f"tamper_detection_font_lib_size {m['font_lib_size']}",
        f"tamper_detection_font_lib_ready {1 if m['font_lib_ready'] else 0}",
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; charset=utf-8")


@api_router_v3.get("/health", summary="系统健康检查")
async def health_check(engine: InferenceEngineAPI = Depends(get_engine)):
    return {
        "status": "ok",
        "font_lib_ready": engine.font_lib.is_ready,
        "font_lib_size": engine.font_lib.index.ntotal,
        "global_model_loaded": engine.global_model is not None,
        "ocr_available": engine.extractor.reader is not None,
    }


app.include_router(api_router_v3)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8030, workers=1)
