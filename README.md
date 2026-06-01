# 算式求解器项目 README

本项目是 EE326 数字图像处理课程 Project，目标是用传统数字图像处理方法从算式图片中识别表达式，并输出可计算的表达式或计算结果。当前项目重点在图像预处理、字符/结构分割、模板匹配、规则识别、样例数据制作和标签管理，不使用机器学习、深度学习、OCR 模型或在线识别 API。

## 当前状态

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 分类算式识别 | 已完成主要功能 | 对已知类别分别调用对应识别流程，当前覆盖印刷体四则运算、负数/小数、二维结构、基础手写符号、微积分样例。 |
| 数据和标签制作 | 已完成主要功能 | 已制作 `data/samples/` 下多类样例图片，并在 `data/labels/` 下提供同名 `.txt` 标签。 |
| 无分类算式/算子识别 | 已完成主要改进 (2026-06-01) | Auto router 已从硬阈值规则链重构为 4 层打分+交叉验证架构，整体 auto 准确率 94.9%。详见下方"Auto Router 改进"章节。 |
| UI 界面 | 已完成基本功能 (2026-06-01) | Gradio Web UI：拖拽上传、模式切换、结果/Token 展示、处理过程可视化。见下方"Web UI"章节。 |
| UI 接入计算 API 或其他计算方法 | 部分完成 | 本地 ArithmeticSolver / SymbolicSolver 已接入 UI。远程 API、批量处理、WebSocket 尚未实现。 |

## 项目树

下面只列主要文件和目录，省略 `__pycache__`、临时调试图片和部分重复样例文件。

```text
Project/
├── README.md
├── PROJECT_CONTEXT.md
├── ITERATION_LOG.md
├── requirements.txt
├── run.py
├── evaluation_results*.csv
├── 讲解.html
├── 解释.html
├── docs/
│   └── superpowers/
│       ├── specs/
│       │   └── 2026-06-01-auto-router-improvements-design.md
│       └── plans/
│           └── 2026-06-01-auto-router-improvements-plan.md
├── data/
│   ├── samples/
│   │   ├── printed_basic/
│   │   ├── printed_decimal_negative/
│   │   ├── printed_2d_layout/
│   │   ├── handwritten_basic/
│   │   ├── calculus/
│   │   └── test/
│   ├── labels/
│   │   ├── printed_basic/
│   │   ├── printed_decimal_negative/
│   │   ├── printed_2d_layout/
│   │   ├── handwritten_basic/
│   │   ├── calculus/
│   │   └── test/
│   └── templates/
│       └── printed_basic/
├── src/
│   ├── expression/
│   │   ├── calculus.py
│   │   ├── display_normalizer.py
│   │   ├── normalizer.py
│   │   ├── serializer.py
│   │   └── types.py
│   ├── solver/
│   │   ├── arithmetic.py
│   │   ├── base.py
│   │   └── symbolic.py
│   └── vision/
│       ├── auto_router.py
│       ├── calculus_layout.py
│       ├── calculus_rules.py
│       ├── deskew.py
│       ├── layout_analysis.py
│       ├── line_segmentation.py
│       ├── normalization.py
│       ├── pipeline.py
│       ├── preprocess.py
│       ├── segmentation.py
│       ├── structure_2d.py
│       ├── structure_router.py
│       ├── symbol_segmentation.py
│       ├── template_matcher.py
│       ├── types.py
│       └── recognizers/
│           ├── handwritten_rule_template.py
│           └── printed_template.py
├── tools/
│   ├── evaluate_samples.py
│   └── generate_printed_templates.py
├── ui/
│   ├── __init__.py
│   └── app.py
├── tests/
│   ├── test_arithmetic_solver.py
│   ├── test_auto_router.py
│   ├── test_calculus_layout.py
│   ├── test_calculus_rules.py
│   ├── test_cross_validation.py
│   ├── test_disambiguation_layers.py
│   ├── test_handwritten_leave_one_out.py
│   ├── test_structure_2d.py
│   ├── test_structure_router_scoring.py
│   └── ...
└── debug/
    └── ...
```

## 已完成的功能

### 1. 分类可实现的算式识别

项目已经按类别实现多条识别链路。使用时可以明确指定识别器：

```powershell
python run.py data\samples\printed_basic\printed_basic_001.png --recognizer printed
python run.py data\samples\handwritten_basic\handwritten_3_001.png --recognizer handwritten
python run.py data\samples\calculus\calculus_001.png --recognizer calculus --backend symbolic
```

注意：`run.py` 当前提供的命令行识别器选项是 `printed`、`handwritten`、`calculus` 和 `auto`。`printed_2d_layout` 的二维结构识别模块已经实现，但没有作为 `run.py --recognizer` 的独立选项暴露；单张二维结构图片建议通过 `auto` 路由运行，或通过评估脚本的 `--category printed_2d_layout` 批量验证。

```powershell
python run.py data\samples\printed_2d_layout\printed_2d_001.png --recognizer auto
python tools\evaluate_samples.py --category printed_2d_layout --output evaluation_results_2d.csv
```

当前分类识别覆盖：

- `printed_basic`：印刷体基础四则运算。
- `printed_decimal_negative`：含负数、小数、乘除符号规范化的印刷体表达式。
- `printed_2d_layout`：分数、上标、下标、根号等二维排版结构。
- `handwritten_basic`：基础手写单字符/符号识别，使用规则和模板匹配，不训练模型。
- `calculus`：规范样例中的导数、积分、极限结构识别，并转换为符号计算可用文本。

主要处理流程包括：

1. 图像读取、灰度化、二值化、归一化和必要的倾斜校正。
2. 基于投影、连通域、轮廓和几何规则进行字符或结构分割。
3. 使用固定模板匹配和几何规则完成字符/符号识别。
4. 将识别结果整理成表达式 token、layout AST 或 SymPy 风格文本。
5. 通过本地 `ArithmeticSolver` 或 `SymbolicSolver` 输出计算结果。

### 2. 数据和标签制作

项目已经建立了样例图片和标签目录。当前数据规模如下：

| 类别 | 图片数量 | 标签数量 | 说明 |
| --- | ---: | ---: | --- |
| `printed_basic` | 20 | 20 | 基础印刷体四则运算。 |
| `printed_decimal_negative` | 20 | 20 | 负数、小数、乘除显示规范样例。 |
| `printed_2d_layout` | 20 | 20 | 分数、上下标、根号等二维结构样例。 |
| `handwritten_basic` | 105 | 105 | 21 类基础手写字符，每类 5 张。 |
| `calculus` | 10 | 10 | 导数、积分、极限样例。 |
| `test` | 1 | 1 | 手写诊断样例。 |

标签文件与图片文件同名，例如：

```text
data/samples/printed_basic/printed_basic_001.png
data/labels/printed_basic/printed_basic_001.txt
```

这些数据主要用于分类评估、回归测试和后续方法改进。

### 3. 评估和测试工具

项目提供批量评估脚本：

```powershell
python tools\evaluate_samples.py --category all --output evaluation_results.csv
python tools\evaluate_samples.py --category auto --output evaluation_results_auto.csv
python tools\evaluate_samples.py --category printed_basic --augment-morphology --output evaluation_results_printed_basic_morph.csv
```

支持的评估类别包括：

- `all`
- `auto`
- `printed_basic`
- `printed_decimal_negative`
- `printed_2d_layout`
- `handwritten_basic`
- `calculus`

评估脚本还支持：

- `--debug-failures`：输出失败样例调试信息。
- `--disable-fallbacks`：关闭部分 fallback，用于消融对比。
- `--augment-morphology`：生成腐蚀/膨胀版本，测试形态学扰动鲁棒性。
- `--solver-timeout-ms`：控制求解后端超时时间。
- `--confusion-matrix`：输出每类别的路由去向混淆矩阵（stdout + CSV）。

测试目录 `tests/` 覆盖了预处理、分割、模板匹配、二维结构、微积分结构、表达式序列化、求解器、auto router、交叉验证和符号消歧等模块。

当前已验证结果：

```text
python -m unittest discover -s tests
Ran 76 tests
OK (skipped=1)
```

## Auto Router 改进 (2026-06-01)

Auto router 已从硬阈值 if-else 规则链重构为 **4 层流水线架构**。设计文档见 `docs/superpowers/specs/2026-06-01-auto-router-improvements-design.md`。

### Layer 1：增强特征提取 (`src/vision/structure_router.py`)

`ImageStructureFeatures` 从原来的 10 个字段扩展到 16 个字段，新增：

| 新特征 | 用途 |
|---|---|
| `vertical_symmetry` | 上下半部分前景对称性（印刷体对称，手写不对称） |
| `component_height_variance` | 各连通分量高度的变异系数（手写高低不一） |
| `horizontal_run_count` | 水平投影行数（检测上下标多行结构） |
| `left_right_density_ratio` | 左右密度比（积分符号左侧偏重） |
| `multi_scale_edge_ratio` | 双阈值 Canny 边缘比例（印刷体≈1，手写>1.4） |
| `top_alignment_score` | 分量顶部对齐程度（印刷体对齐，手写漂移） |

新增 4 个派生属性：`is_handwritten_texture`、`has_integral_structure`、`is_grid_aligned`、`is_vertically_symmetric`。

### Layer 2：五类别并行打分 (`src/vision/structure_router.py`)

5 个评分函数（`_score_printed_basic`、`_score_printed_2d_layout`、`_score_handwritten_basic`、`_score_calculus`、`_score_printed_decimal_negative`）同时对同一张图片打分。每个函数从基线分出发，累加特征权重贡献。取最高分作为路由决策。当第一名与第二名边距 < 0.12 时降低置信度，触发 Layer 3 交叉验证。

### Layer 3：双路并行交叉验证 (`src/vision/auto_router.py`)

当路由置信度 ≤ 0.7 时，同时运行前两名候选类别的识别器，比较 token 重叠率（Jaccard）、布局类型一致性和求解器答案，综合得分更高的候选胜出。

### Layer 4：三层符号消歧 (`src/vision/recognizers/handwritten_rule_template.py`)

- **Layer A** — 硬几何规则：2 个洞 → "8"，3 分量 + 正方形 → "÷"，宽高比 > 2.5 + 水平线 → "-"
- **Layer B** — 上下文感知：根据 `context` 参数（`"handwritten"` / `"calculus"`）调整候选分数。微积分模式下提升 `d`、`∫`、`x`，惩罚易混淆数字
- **Layer C** — 混淆对规则（新增 7 组）：`0`/`o`/`O`、`,`/`.`、`2`/`z`、`5`/`s`、`9`/`g`、`6`/`b`、`-`/`_`

### 配套增强

- **微积分符号覆盖** (`src/vision/calculus_layout.py`)：新增 `d` 和 `0` 误识别恢复规则
- **函数名前缀映射** (`src/vision/calculus_rules.py`)：扩展 `arcsin/arccos/arctan` 的 OCR 错误模式
- **混淆矩阵输出** (`tools/evaluate_samples.py`)：`--confusion-matrix` 输出每类别路由去向

### 效果

| 类别 | 改进前 | 改进后 |
|---|---|---|
| calculus | 10/10 | 10/10 |
| handwritten_basic | 102/105 | 102/105 |
| printed_2d_layout | 10/20 | **14/20** |
| printed_basic | 20/20 | 20/20 |
| printed_decimal_negative | 20/20 | 20/20 |
| **Overall** | **162/175 (92.6%)** | **166/175 (94.9%)** |

### 新增测试文件

| 文件 | 覆盖内容 |
|---|---|
| `tests/test_structure_router_scoring.py` | 5 个评分函数在典型特征向量上的排序正确性 |
| `tests/test_cross_validation.py` | `_cross_validate()` 单元测试（一致/矛盾/部分重叠/空候选） |
| `tests/test_disambiguation_layers.py` | Layer A/B/C 规则不冲突，每个混淆对的区分逻辑 |

## Web UI

项目已接入 Gradio 构建的 Web 界面，提供图形化操作和实时识别。

### 启动方式

```powershell
python ui\app.py
```

启动后访问 `http://127.0.0.1:7861`。

### 界面功能

| 功能 | 说明 |
|---|---|
| 图片上传 | 支持拖拽或点击上传算式图片，也支持剪贴板粘贴 |
| 识别模式切换 | Radio 按钮选择 `printed` / `handwritten` / `calculus` / `auto` |
| 计算后端切换 | `auto`（自动选择）/ `arithmetic`（四则运算）/ `symbolic`（符号计算） |
| 结果展示 | 显示识别表达式、计算结果和错误信息 |
| Token 列表 | 表格展示每个 token 的符号、类型、置信度和边界框 |
| 处理过程可视化 | 展示二值化结果和字符分割框 |
| 结构分析 | Auto 模式下显示路由决策理由、候选分数和图像特征详情 |
| 警告信息 | 显示低置信度 token 和未知符号警告 |

### API 访问

Gradio 界面自动提供 REST API 端点，可通过编程方式调用：

```powershell
curl http://127.0.0.1:7861/api/
```

## 待完成功能：远程计算 API 接入

当前已有本地计算：

- `ArithmeticSolver`：处理普通四则运算。
- `SymbolicSolver`：基于本地 `sympy` 处理部分符号计算。

尚未实现：

- 接入远程计算 API（如 Wolfram Alpha、Mathpix 等）
- 批量图片处理 API endpoint
- WebSocket 实时识别流

## 安装依赖

依赖记录在 `requirements.txt`：

```text
numpy>=1.20.0
opencv-python>=4.5.0
sympy>=1.10
Pillow>=9.0.0
gradio>=4.0.0
```

安装方式：

```powershell
pip install -r requirements.txt
```

其中 `Pillow` 主要用于 `tools/generate_printed_templates.py` 生成印刷体模板图片；`gradio` 用于 Web UI；核心识别流程主要依赖 OpenCV、NumPy 和 SymPy。

## 常用命令

运行单张图片识别：

```powershell
python run.py data\samples\printed_basic\printed_basic_001.png --recognizer printed
```

运行自动识别原型：

```powershell
python run.py data\samples\test\1.jpg --recognizer auto
```

输出调试文件：

```powershell
python run.py data\samples\printed_basic\printed_basic_001.png --recognizer printed --debug-dir debug\demo
```

运行全量分类评估：

```powershell
python tools\evaluate_samples.py --category all --output evaluation_results.csv
```

运行单元测试：

```powershell
python -m unittest discover -s tests
```

启动 Web UI：

```powershell
python ui\app.py
```

## 项目约束

- 图像识别部分不使用机器学习、深度学习、OCR 模型或在线识别 API。
- 允许使用 OpenCV、NumPy 等传统图像处理工具。
- 允许在表达式已经识别出来之后，使用本地符号计算库或后续 API 做计算，但需要明确区分“识别”和“计算”的职责。
- 课程展示重点应放在图片到表达式的转换过程，包括预处理、分割、模板匹配、结构分析和规则判断。
