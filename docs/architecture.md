# 架构说明

识别主流程：

```text
image
  -> normalize/preprocess
  -> segment
  -> recognize tokens
  -> build layout
  -> serialize expression
  -> solve
```

## 入口

- `run.py`：命令行入口。
- `ui/app.py`：Gradio Web UI。
- `src/vision/pipeline.py`：识别器分发入口。
- `tools/evaluate_samples.py`：样例集批量评估。

## 识别器

`src/vision/pipeline.py` 支持四种后端：

- `printed`：`PrintedTemplateRecognizer`，用于印刷体基础算式。
- `handwritten`：`HandwrittenRuleTemplateRecognizer`，用于基础手写符号。
- `calculus`：`recognize_calculus_layout`，用于导数、积分、极限样例。
- `auto`：`recognize_unknown`，先分析图像结构，再选择候选识别器。

`printed_2d_layout` 没有作为 `run.py --recognizer` 的独立选项暴露；单张二维结构图片建议用 `--recognizer auto`，批量验证用评估脚本的 `--category printed_2d_layout`。

## 核心模块

- `src/vision/preprocess.py`：读图、灰度化、二值化和形态学处理。
- `src/vision/normalization.py`：裁剪、缩放和版面归一化。
- `src/vision/segmentation.py`：字符级分割。
- `src/vision/template_matcher.py`：模板读取、特征提取和匹配。
- `src/vision/layout_analysis.py`：基础二维结构分析。
- `src/vision/structure_2d.py`：分数、上下标、根号等二维排版识别。
- `src/vision/calculus_layout.py`：微积分结构识别。
- `src/vision/auto_router.py`：未知类别输入的候选识别和选择。
- `src/vision/structure_router.py`：图像结构特征提取和类别打分。

## 表达式与求解

- `src/expression/types.py`：`SymbolToken`、`RecognitionResult`、`LayoutNode` 等核心数据结构。
- `src/expression/normalizer.py`：token 到表达式文本的规范化。
- `src/expression/serializer.py`：layout AST 到 plain text 或 SymPy 文本的序列化。
- `src/solver/arithmetic.py`：本地四则运算解析和求值。
- `src/solver/symbolic.py`：基于 SymPy 的符号求解。

## Auto Router

Auto router 的设计目标是处理未知类别输入：

1. 提取结构特征，例如连通域数量、长水平线、分数线、堆叠组件、边缘粗糙度和左右密度。
2. 对 `printed_basic`、`printed_decimal_negative`、`printed_2d_layout`、`handwritten_basic`、`calculus` 并行打分。
3. 在路由置信度较低时运行候选识别器做交叉验证。
4. 用识别置信度、结构复杂度和语义验证结果选择最终候选。

详细设计见 `docs/superpowers/specs/2026-06-01-auto-router-improvements-design.md`。
