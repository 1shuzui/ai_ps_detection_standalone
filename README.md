# AI 图像鉴伪系统 — 独立版

基于深度学习的图像篡改检测系统，针对**证件、票据、单据类图片**中的关键数值区域（金额、单号、日期等），通过全局布局 + 局部像素 + 字体风格三重信号交叉验证，判断是否存在 PS/拼接/生成式篡改。

## 目录

- [快速开始](#快速开始)
- [架构概览](#架构概览)
- [目录结构](#目录结构)
- [检测原理](#检测原理)
- [配置说明](#配置说明)
- [API 参考](#api-参考)
- [人工反馈与模型训练](#人工反馈与模型训练)
- [调优指南](#调优指南)

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.x（可选，GPU 加速推理）
- 内存 ≥ 8GB，显存 ≥ 4GB（GPU 模式）

### 安装

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. 安装依赖
pip install -r requirements.txt
```

首次运行时会自动下载 EasyOCR 模型（约 200MB）。

### 启动服务

```bash
python main.py
```

服务默认监听 `http://0.0.0.0:7000`。启动后访问 `http://localhost:7000/docs` 查看 Swagger 文档。

### 快速测试

```bash
# 单图推理测试
python inference_api.py
```

## 架构概览

```
请求 → FastAPI 路由层 (main.py)
           │
           ├─ 同步接口 /api/v1  ─┐
           ├─ 异步接口 /api/v3  ─┤
           └─ 反馈/训练接口      ─┤
                                  ▼
                    InferenceEngineAPI (inference_api.py)
                      │         │          │
                      ▼         ▼          ▼
                 全局特征    局部特征    EXIF分析
               (ResNet-18)  (OCR+FAISS)  (元数据)
                      │         │          │
                      └─────────┼──────────┘
                                ▼
                     融合策略 (max + 一致信号加成)
                                │
                                ▼
                    结果: 正常 / 可疑 / 篡改
```

### 三条检测通路

| 通路 | 技术 | 检测目标 |
|------|------|----------|
| **全局布局** | ResNet-18 512维特征 + XGBoost | 整图 UI 布局是否被破坏、拼接痕迹 |
| **局部像素** | ELA + 拉普拉斯 + 噪声/颜色一致性 | 像素级涂抹、复制粘贴、生成器假图 |
| **字体风格** | EasyOCR + FAISS 字体库比对 | 关键数字字体是否与正版字体一致 |

三条通路的结果通过**融合策略**综合判定，而非简单取最大值——多信号一致时会获得置信度加成。

### 为什么是三重信号？

单一信号容易误判。例如：
- 全局布局异常可能是原始图片质量问题，不一定是篡改
- 像素异常可能是 JPEG 压缩伪影
- 字体异常可能是因为该字体样本不足

三条信号相互印证时，置信度大幅提升；只有单条信号时保持克制，降低误报。

## 目录结构

```
ai_detection_standalone/
├── main.py                  # FastAPI 服务入口，路由定义
├── inference_api.py         # 推理引擎封装（核心检测逻辑）
├── config.yaml              # 所有可调参数集中管理
├── requirements.txt         # Python 依赖
├── feedback_manager.py      # 人工标注反馈管理
├── train_pipeline_v2.py     # 模型训练管线
├── train_pipeline.py        # 旧版训练脚本（保留兼容）
├── batch_eval_amounts.py    # 批量评估脚本 + OCR 候选框辅助函数
│
├── core/                    # 核心算法模块
│   ├── detectors.py         # 像素检测器 + EXIF 原始性分析
│   ├── extractors.py        # 特征提取器 + 字体库 + 排版一致性分析
│   ├── augmentations.py     # 数据增强（训练用）
│   └── utils.py             # 工具函数（字体加载、安全读图、JSON 编码）
│
├── models/                  # 模型文件
│   ├── font_lib.index       # FAISS 字体特征库
│   ├── font_lib_meta.pkl    # 字体库元数据
│   ├── global_layout_model.pkl  # XGBoost 全局布局模型
│   └── trained/             # 训练产出（模型版本 + 可视化）
│       └── viz/             # 训练可视化图片
│
├── images/                  # 训练用正负样本图片
├── locate_json/             # 训练样本的标注 JSON
├── pptest/                  # 测试用图片
├── feedback/                # 人工标注反馈归档
│   ├── correct/             # 判定正确的样本
│   ├── wrong/               # 判定错误的样本（含原图+裁剪+元数据）
│   └── suspicious/          # 疑似错误，待确认
│
├── static/                  # 前端静态资源
└── results/                 # 批量评估结果
```

## 检测原理

### 1. 全局特征分析

```
图片 → ResNet-18 (去掉分类头) → 512维特征向量 → XGBoost 分类器 → 全局篡改概率
```

- 使用预训练 ResNet-18 提取图片的深层语义特征
- XGBoost 在标注数据集上训练，学习"被篡改过的 UI 布局"的模式
- 对整体拼接、复制粘贴等破坏排版一致性的操作敏感

### 2. 局部像素分析（PixelLevelDetector）

逐个 ROI 区域进行五层检测：

| 检测层 | 方法 | 捕获的篡改类型 |
|--------|------|---------------|
| ELA（误差水平分析） | JPEG 重压缩后像素差异 | 拼接区域的重压缩伪影 |
| 拉普拉斯边缘 | 边缘梯度突变统计 | 粘贴边界的锐利过渡 |
| 生成器假图 | 背景方差异常低 | GAN/扩散模型生成的纯色背景 |
| 噪声一致性 | ROI vs 周围背景的噪声方差对比 | 粘贴区域噪声模式不一致 |
| 颜色一致性 | 相邻 ROI 的颜色直方图对比 | 从不同图片拼凑的颜色差异 |

### 3. 字体风格分析（FontFeatureLibrary）

```
ROI → EasyOCR 定位文字 → 筛选含数字的区域 → ResNet-18 提取字形特征
                                                 │
                                                 ▼
                                          FAISS 向量库检索
                                                 │
                                     L2 距离 → 相似度 (指数衰减)
```

- **只对含数字的区域**提取字体特征（避免"净重""吨"等纯中文干扰）
- 字体库从正版样本中构建，包含各种字号、光照条件下的正版字体
- 相似度通过校准后的指数衰减函数映射到 [0,1]

### 4. 排版一致性（TamperAnalyzer）

对同一行内的数字区域检查：

- **高度一致性**：同行的数字高度方差不应过大（防大小字拼接）
- **基线对齐**：同行的 Y 坐标方差不应超过 15px（防错位拼接）

### 5. EXIF 元数据分析（OriginalityChecker）

- 检测图片元数据中是否包含已知修图软件（Photoshop、美图秀秀等）的特征
- 分析噪声模式、颜色熵等图像统计特征

### 6. 融合策略

```
final_risk = max(全局概率, 局部概率, EXIF风险)

如果 2 个以上信号同时报警：
    final_risk += 0.08 ~ 0.12   (一致信号加成)

final_risk = max(final_risk, 加权平均 × 0.7)  (加权平均值作为参考下限)
```

融合方法可在 `config.yaml` 的 `fusion.method` 切换：
- `"weighted"`（推荐）：max 保底 + 一致加成 + 加权参考
- `"max"`：纯最大值

## 配置说明

所有参数集中在 `config.yaml`，按功能分为以下配置节：

### business_rules — 业务规则

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `roi_expand_margin` | 15 | ROI 外扩像素数，用于获取周围背景做噪声对比 |
| `max_core_text_length` | 15 | 超过此长度的文本被视为非核心字段（如长订单号） |

### weights — 自适应权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `core_pixel` | 0.60 | 核心字段（金额等）：像素异常权重 |
| `core_font` | 0.40 | 核心字段：字体异常权重 |
| `non_core_pixel` | 0.80 | 非核心字段：纯像素异常权重（字体信号不可用时） |

核心字段 vs 非核心字段的判断逻辑：文本中 ≥3 个数字且数字占比 ≥35%，或包含金额/单号关键词。

### thresholds — 阈值控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `global_fake` | 0.65 | 全局模型输出 > 此值视为排版异常 |
| `pixel_anomaly_alert` | 0.60 | 像素异常 > 此值触发拼接/涂抹报警 |
| `exempt_pixel_safe` | 0.40 | 非核心字段像素异常 < 此值直接清零（降低误报） |
| `suspect_high` | 0.65 | 综合风险 > 此值判定为"篡改" |
| `suspect_low` | 0.50 | 综合风险 > 此值判定为"可疑"，≤ 则为"正常" |

### fusion — 融合策略

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `method` | `"weighted"` | 融合方法：`"weighted"` 或 `"max"` |
| `weight_global` | 0.35 | 加权融合中全局特征的权重 |
| `weight_local` | 0.65 | 加权融合中局部特征的权重 |

### pixel_detector — 像素检测器

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `generator_enabled` | true | 是否启用生成器假图检测 |
| `generator_bg_var_threshold` | 0.05 | 背景方差低于此值触发生成器嫌疑 |
| `generator_penalty` | 0.70 | 生成器假图的惩罚分（可调低减少白底文档误报） |
| `noise_consistency_weight` | 0.15 | 噪声一致性在像素异常中的权重 |
| `color_consistency_weight` | 0.10 | 颜色一致性在像素异常中的权重 |

### originality — EXIF 分析

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | true | 是否启用 EXIF/元数据分析 |

### feedback — 反馈系统

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | true | 是否启用反馈系统 |
| `storage_dir` | `"feedback"` | 标注数据存储根目录 |

### training — 训练系统

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | `"models/trained"` | 训练产出目录 |
| `visualization_enabled` | true | 是否生成训练可视化图 |
| `visualization_dir` | `"models/trained/viz"` | 可视化图保存路径 |
| `backup_previous` | true | 训练前自动备份旧模型 |

## API 参考

所有接口的完整文档见 `http://localhost:7000/docs`（Swagger UI）。

### 同步接口（单图单框）

```
POST /api/v1/image-detection/detect
Content-Type: multipart/form-data

file: <图片文件>
bbox: "[120, 80, 400, 140]"   # JSON 数组或逗号分隔
```

响应示例：

```json
{
  "status": "success",
  "data": {
    "result": "可疑",
    "confidence": 0.58,
    "bbox": [120, 80, 280, 60],
    "reason": "存在局部边缘拼接/像素涂抹痕迹"
  }
}
```

### 异步接口（单图多框 / 自动扫描）★ 推荐

**提交任务：**

```
POST /api/v3/detect
Content-Type: multipart/form-data

file: <图片文件>
bbox: <可选，不传则自动 OCR 多框检测>
```

**查询结果：**

```
GET /api/v3/result/{task_id}
```

**获取可视化：**

```
GET /api/v3/result/{task_id}/visualization
```

返回带检测框和风险标签标注的图片。

**取消任务：**

```
DELETE /api/v3/task/{task_id}
```

### 反馈接口

```
POST   /api/v3/feedback/judge      # 提交人工标注 (correct/wrong/suspicious)
GET    /api/v3/feedback/list       # 列出反馈记录，支持 ?judgment=wrong 过滤
POST   /api/v3/feedback/confirm    # 确认疑似标注转向
```

### 训练接口

```
POST   /api/v3/train               # 触发训练（必须先传 confirm=true）
GET    /api/v3/train/viz/{filename}  # 获取训练可视化图片
```

### 检测历史

```
GET    /api/v1/history                    # 分页查询历史记录
GET    /api/v1/history/{record_id}/image  # 获取历史图片
```

## 人工反馈与模型训练

### 反馈闭环

```
检测 → 人工审核 → 标注 (correct/wrong/suspicious) → 训练 → 模型升级
  ↑                                                                    │
  └────────────────────────────────────────────────────────────────────┘
```

1. 通过 `/api/v3/feedback/judge` 提交标注
   - `wrong`：系统自动保存原图 + OCR 框选区域裁剪图 + 完整元数据
   - `suspicious`：暂存到待确认目录，通过 `/api/v3/feedback/confirm` 二次确认
   - `correct`：保存为正常样本
2. 积累一定量反馈数据后，调用 `/api/v3/train` 启动训练
3. 训练前系统会**自动备份旧模型**到 `models/trained/backup_xxx/`
4. 训练产出：新 XGBoost 模型 + 新字体库 + 训练可视化图片

### 训练数据来源

| 来源 | 标签 | 用途 |
|------|------|------|
| `images/` 目录（文件名含 "no" 的） | 正样本 (label=0) | 全局模型 + 字体库 |
| `images/` 目录（其他文件） | 负样本 (label=1) | 全局模型 |
| `feedback/wrong/` | 负样本 (label=1) | 全局模型增强 |

### 可视化输出

训练完成后在 `models/trained/viz/` 生成三张图：
1. `feature_importance_xxx.png` — XGBoost 特征重要性 Top 30
2. `training_analysis_xxx.png` — 分数分布直方图 + 混淆矩阵
3. `learning_curve_xxx.png` — 不同训练数据量下的准确率曲线

## 调优指南

### 场景 1：白底文档误报偏高

白底文档容易触发生成器假图检测（纯白背景方差过低）。

**调整方案**：
```yaml
pixel_detector:
  generator_penalty: 0.40    # 降低生成器惩罚分（默认 0.70）
  # 或直接关闭：
  generator_enabled: false
```

### 场景 2：希望提高召回率（宁可多报不漏报）

**调整方案**：
```yaml
thresholds:
  suspect_low: 0.40           # 降低"可疑"门槛
  suspect_high: 0.55          # 降低"篡改"门槛
  pixel_anomaly_alert: 0.45   # 降低像素报警门槛
```

### 场景 3：希望降低误报率（宁可不报也不错报）

**调整方案**：
```yaml
thresholds:
  suspect_low: 0.60
  suspect_high: 0.75
  global_fake: 0.75
  exempt_pixel_safe: 0.55     # 提高豁免门槛

fusion:
  method: "max"               # 关掉一致信号加成，纯 max
```

### 场景 4：字体库不足导致字体信号不稳定

如果检测的票据字体与字体库差异较大（如手写票据、特殊字体），字体信号可能不可靠。

**调整方案**：
```yaml
weights:
  core_font: 0.20             # 降低字体权重
  core_pixel: 0.80            # 像素权重相应提高
```

长期方案：收集更多正版样本并通过训练更新字体库。

### 场景 5：GPU 显存不足

```bash
# 环境变量控制
export CUDA_VISIBLE_DEVICES=""     # 强制 CPU 模式
export TORCH_NUM_THREADS=4         # 限制 CPU 线程数
```

或在 `main.py` 中调整 `MAX_CONCURRENT_AI_TASKS = 1`（已经是默认最保守值）。

### 调试模式

```bash
# 单图详细推理日志
python inference_api.py

# 批量评估（查看每条检测的详细分数）
python batch_eval_amounts.py
```

批量评估结果输出到 `results/` 目录，包含每张图的分类统计和混淆矩阵。

## 常见问题

**Q: 首次启动很慢？**
A: EasyOCR 首次运行需要下载模型文件（约 200MB），之后会缓存到本地。建议首次在稳定网络环境下启动。

**Q: 如何判断检测结果是否可靠？**
A: 看 `confidence` 和 `reason` 字段。confidence > 0.65 的"篡改"判定可信度较高；confidence 在 0.50-0.65 的"可疑"建议人工复核。reason 中同时包含"全局UI布局异常"和"局部像素痕迹"时比单独一条更可信。

**Q: 训练后效果变差了怎么办？**
A: 训练前会**自动备份旧模型**到 `models/trained/backup_xxx/` 目录。手动将备份文件复制回 `models/` 即可回滚。

**Q: 支持哪些图片格式？**
A: JPG、JPEG、PNG。建议使用原始分辨率图片，不要过度压缩（ELA 依赖 JPEG 压缩特征）。
