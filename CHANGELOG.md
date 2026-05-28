# Changelog

## 2026-05-26 — P0-P2 完善 (CPU-only)

### Phase 1: 基础层

#### P2.9 提取 OCR 工具到 `core/ocr_utils.py`
- **新建** [core/ocr_utils.py](core/ocr_utils.py) — 从 `batch_eval_amounts.py` 提取共享 OCR 工具（OCRToken、AmountCandidate 数据类、正则常量、评分函数、候选框构建、凭证检测）
- **修改** [main.py:35](main.py#L35) — import 路径从 `batch_eval_amounts` 改为 `core.ocr_utils`
- **修改** [batch_eval_amounts.py](batch_eval_amounts.py) — 删除 ~460 行本地重复代码，改为从 `core.ocr_utils` 导入
- 解决 `main.py` 从评估脚本导入核心域逻辑的语义问题

#### P2.10 异常分类 `core/exceptions.py`
- **新建** [core/exceptions.py](core/exceptions.py) — DetectionError（基类）、RecoverableError（用户可恢复）、SystemError（需运维）、ImageReadError、ModelInferenceError
- **修改** [core/detectors.py:94](core/detectors.py#L94) — `_check_noise_consistency` 的 `except Exception` 改为 `except cv2.error`
- **修改** [core/detectors.py:112](core/detectors.py#L112) — `_check_color_consistency` 的 `except Exception` 改为 `except cv2.error`
- **修改** [feedback_manager.py:87](feedback_manager.py#L87) — `except Exception: pass` 改为 `except (cv2.error, OSError): pass`
- **修改** [inference_api.py:276](inference_api.py#L276) — catch-all 区分 RecoverableError（用户友好提示）与系统异常（联系运维）

#### P0.2 配置集中化
- **修改** [config.yaml](config.yaml) — 新增 `server`（storage_dir/max_concurrent_tasks/gc）、`ocr`（contrast/mag_ratio/threshold）、`preprocessing`（preserve_aspect_ratio）配置节
- **修改** [main.py:45-51](main.py#L45) — STORAGE_DIR、并发数、GC 参数从硬编码改为读取 config，默认 `data/storage`（非 /tmp）
- **修改** [main.py](main.py) — OCR 参数从 config 读取，`_run_ocr_once` 不再硬编码
- **修改** [.gitignore](.gitignore) — 添加 `data/`

### Phase 2: 检测增强

#### P1.4 字体库冷启动降级
- **修改** [core/extractors.py](core/extractors.py) — FontFeatureLibrary 新增 `is_ready` 属性（`index.ntotal > 0`）
- **修改** [inference_api.py](inference_api.py) — `predict()` 中字体库为空时强制 `should_use_font_signal = False`，跳过 FAISS 查询，像素权重自动提升
- **修改** [main.py](main.py) — 新增 `GET /api/v3/health` 端点，返回字体库状态、模型加载、OCR 可用性

#### P1.5 宽高比感知预处理 PadToSquare
- **修改** [core/extractors.py](core/extractors.py) — 新增 `PadToSquare` 变换类（边缘复制填充），`FeatureExtractor` 支持 `preserve_aspect_ratio` 参数
- **修改** [inference_api.py](inference_api.py) — `FeatureExtractor` 初始化时从 config 读取 `preprocessing.preserve_aspect_ratio`
- **修改** [config.yaml](config.yaml) — `preprocessing.preserve_aspect_ratio` 控制开关（默认 true）

#### P1.7 DCT 频域分析（生成式 AI 检测）
- **修改** [core/detectors.py](core/detectors.py) — `PixelLevelDetector` 新增 `_check_dct_anomaly()` 方法：8x8 分块 DCT + 高频系数分布分析
- **修改** [config.yaml](config.yaml) — `pixel_detector` 新增 `dct_analysis_enabled`、`dct_weight`
- CPU 友好：纯 numpy/cv2 运算，无 GPU 依赖

#### P1.6 跨 ROI 一致性分析
- **修改** [core/extractors.py](core/extractors.py) — `TamperAnalyzer` 新增 `check_cross_roi_consistency()` 静态方法，比较多个 ROI 间的字体高度、基线对齐
- 可用于多框检测路径中的跨区域一致性校验

### Phase 3: 架构改进

#### P2.11 依赖注入容器 `core/app_context.py`
- **新建** [core/app_context.py](core/app_context.py) — `AppContext` 类封装 engine、registry、ocr_reader、semaphore，提供 `initialize()`/`shutdown()` 异步方法
- 与 EngineContainer 共存，后续版本移除模块级单例

#### P0.3 模型热加载
- **修改** [inference_api.py](inference_api.py) — `InferenceEngineAPI` 新增 `reload_models()` 方法，原子替换 FAISS 字体库和 XGBoost 模型
- **修改** [main.py](main.py) — 新增 `POST /api/v3/reload` 端点，训练完成后自动复制产出到活跃路径并调用 reload

### Phase 4: 收尾与测试

#### P2.8 删除旧训练脚本
- **删除** `train_pipeline.py` — 功能已由 `train_pipeline_v2.py` 完全覆盖

#### P0.1 测试套件
- **新建** [tests/](tests/) — 62 个单元测试覆盖 5 个模块：
  - `test_detectors.py` — PixelLevelDetector（ELA/噪声/颜色/DCT）、OriginalityChecker
  - `test_extractors.py` — FontFeatureLibrary（CRUD/冷启动）、TamperAnalyzer（内部/跨ROI一致性）
  - `test_fusion.py` — 融合策略边界（max/weighted/bonus/全0/全1）
  - `test_inference_api.py` — bbox 归一化、文本特征分析
  - `test_ocr_utils.py` — OCRToken 解析、金额评分、候选框构建

### 验证结果
- `python -c "from main import app"` — 导入通过
- `pytest tests/ -v` — 62 passed, 0 failed
