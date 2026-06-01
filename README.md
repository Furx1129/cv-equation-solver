# 算式求解器项目 README

本项目是 EE326 数字图像处理课程 Project，目标是用传统数字图像处理方法从算式图片中识别表达式，并输出可计算的表达式或计算结果。当前项目重点在图像预处理、字符/结构分割、模板匹配、规则识别、样例数据制作和标签管理，不使用机器学习、深度学习、OCR 模型或在线识别 API。

## 当前状态

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 分类算式识别 | 已完成主要功能 | 对已知类别分别调用对应识别流程，当前覆盖印刷体四则运算、负数/小数、二维结构、基础手写符号、微积分样例。 |
| 数据和标签制作 | 已完成主要功能 | 已制作 `data/samples/` 下多类样例图片，并在 `data/labels/` 下提供同名 `.txt` 标签。 |
| 无分类算式/算子识别 | 待完善 | 已有 `auto` 路由原型，但仍依赖结构先验和 fallback 规则，当前更适合作为实验版本，后续需要继续修改和增强泛化能力。 |
| UI 界面 | 未完成 | 项目目前没有正式 UI 模块。 |
| UI 接入计算 API 或其他计算方法 | 未完成 | 命令行已能调用本地算术/符号计算后端，但尚未做 UI 层接入，也没有接入外部计算 API。 |

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
├── tests/
│   ├── test_arithmetic_solver.py
│   ├── test_auto_router.py
│   ├── test_calculus_layout.py
│   ├── test_calculus_rules.py
│   ├── test_handwritten_leave_one_out.py
│   ├── test_structure_2d.py
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

测试目录 `tests/` 覆盖了预处理、分割、模板匹配、二维结构、微积分结构、表达式序列化、求解器和 auto router 等模块。

当前已验证结果：

```text
python -m unittest discover -s tests
Ran 57 tests
OK (skipped=1)
```

## 待完善功能：无分类算式/算子识别

项目当前已有 `auto` 路由流程：

```powershell
python run.py data\samples\test\1.jpg --recognizer auto
python tools\evaluate_samples.py --category auto --output evaluation_results_auto.csv
```

它会先分析图像结构特征，再选择可能的识别路线。当前使用的判断信息包括：

- 宽高比和前景填充率。
- 连通域数量。
- 长水平线、分数线和上下堆叠结构。
- 左侧高组件，用于辅助判断积分、极限等结构。
- 笔画粗糙度和低填充率，用于辅助区分手写输入。

但是这部分仍然需要修改和完善，原因是：

- 当前 `auto` 仍然主要依赖手写规则和结构先验，对真实未知图片的泛化能力没有充分验证。
- 一些算子或符号在不同类别中含义接近，例如 `/`、分数线、减号、除号、变量 `x` 和乘号 `x`，仍可能发生混淆。
- fallback 规则对结果影响较大，说明自动识别链路还不够独立稳定。
- 当前测试集规模有限，`auto` 在现有样例上表现好，不等于对任意输入都稳定。

后续建议优先完善：

1. 更严格地区分一维算式、二维排版、单字符手写和微积分结构。
2. 增加真实拍照、不同字体、不同笔画粗细和轻微噪声样例。
3. 将 router 的中间判断结果可视化，方便解释为什么选择某个识别器。
4. 针对容易混淆的算子建立更细的几何规则和置信度校准。

## 未完成功能：UI 与计算 API 接入

目前项目没有正式 UI。也就是说，还没有实现下面这些功能：

- 图形界面选择或拖拽图片。
- 在界面中显示原图、二值化结果、分割结果和识别 token。
- 在界面中显示识别出的表达式、计算结果和错误提示。
- 在 UI 中切换 `printed`、`handwritten`、`calculus`、`auto` 等识别模式。
- 在 UI 中接入计算 API、远程计算服务或其他计算后端。

当前已有的是命令行层面的本地计算：

- `ArithmeticSolver`：处理普通四则运算。
- `SymbolicSolver`：基于本地 `sympy` 处理部分符号计算。

因此，后续 UI/API 工作可以把现有 `run.py` 和 `src/vision/pipeline.py` 作为后端入口，但需要新增界面层和 API 调用层。

## 安装依赖

依赖记录在 `requirements.txt`：

```text
numpy>=1.20.0
opencv-python>=4.5.0
sympy>=1.10
Pillow>=9.0.0
```

安装方式：

```powershell
pip install -r requirements.txt
```

其中 `Pillow` 主要用于 `tools/generate_printed_templates.py` 生成印刷体模板图片；核心识别流程主要依赖 OpenCV、NumPy 和 SymPy。

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

## 项目约束

- 图像识别部分不使用机器学习、深度学习、OCR 模型或在线识别 API。
- 允许使用 OpenCV、NumPy 等传统图像处理工具。
- 允许在表达式已经识别出来之后，使用本地符号计算库或后续 API 做计算，但需要明确区分“识别”和“计算”的职责。
- 课程展示重点应放在图片到表达式的转换过程，包括预处理、分割、模板匹配、结构分析和规则判断。
