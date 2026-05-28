"""
结构化日志配置 — JSON 格式 Formatter，方便接入 ELK/Loki 等日志系统。
"""

import json
import logging
import time
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """将日志记录输出为 JSON 行，包含 timestamp/level/logger/message/module/function。"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def configure_logging(config: dict) -> None:
    """根据 config 配置根日志记录器的格式和级别。"""
    log_cfg = config.get("logging", {})
    use_json = log_cfg.get("json_format", False)
    log_level = log_cfg.get("log_level", "INFO")
    log_file = log_cfg.get("log_file", "")

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 清除已有 handler，避免重复
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    start_time = time.time()
    root_logger.info("日志系统已初始化 json=%s level=%s", use_json, log_level)
    return start_time
