"""
依赖注入容器 — 替代模块级单例 EngineContainer，支持测试和配置隔离。
与 EngineContainer 共存一个版本后移除后者。
"""

import asyncio
import logging
from typing import Any, Optional

from inference_api import InferenceEngineAPI

logger = logging.getLogger(__name__)


class AppContext:
    """封装引擎、注册表、OCR reader 和并发信号量的生命周期管理。"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.engine: Optional[InferenceEngineAPI] = None
        self.registry: Optional[Any] = None
        self.ocr_reader: Optional[Any] = None
        self.ai_semaphore: Optional[asyncio.Semaphore] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self, registry: Any = None) -> None:
        async with self._lock:
            if self._initialized:
                return

            import easyocr
            import torch
            from main import MemoryTaskRegistry

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info("AppContext: 加载 EasyOCR (device=%s)", device)
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=(device == 'cuda'))

            self.engine = InferenceEngineAPI(
                config_path=self.config_path,
                shared_ocr_reader=self.ocr_reader,
            )

            self.registry = registry or MemoryTaskRegistry()

            from main import MAX_CONCURRENT_AI_TASKS
            self.ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AI_TASKS)

            self._initialized = True
            logger.info("AppContext: 初始化完成")

    async def shutdown(self) -> None:
        async with self._lock:
            self.engine = None
            self.registry = None
            self.ocr_reader = None
            self.ai_semaphore = None
            self._initialized = False
            logger.info("AppContext: 已关闭")

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.engine is not None
