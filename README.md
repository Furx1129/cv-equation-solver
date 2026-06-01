# CV Equation Solver

本项目是 EE326 数字图像处理课程 Project：用传统数字图像处理方法从算式图片中识别表达式，并输出可计算表达式或计算结果。

识别部分不使用机器学习、深度学习、OCR 模型或在线识别 API。主要方法包括图像预处理、连通域/投影分割、模板匹配、几何规则、二维结构分析和符号消歧。

## 功能概览

- 印刷体四则运算识别：`printed_basic`
- 负数、小数和乘除符号规范化：`printed_decimal_negative`
- 分数、上下标、根号等二维排版：`printed_2d_layout`
- 基础手写符号识别：`handwritten_basic`
- 导数、积分、极限样例识别：`calculus`
- 无类别输入的自动路由：`auto`
- CLI 和 Gradio Web UI

更多说明见：

- [架构说明](docs/architecture.md)
- [数据集说明](docs/data.md)
- [评估与测试](docs/evaluation.md)
- [未来路线](docs/roadmap.md)
- [Auto Router 改进设计](docs/superpowers/specs/2026-06-01-auto-router-improvements-design.md)

## 安装

```bash
python3 -m pip install -r requirements.txt
```

依赖记录在 `requirements.txt`。核心识别流程主要依赖 OpenCV、NumPy 和 SymPy；Gradio 用于 Web UI。

## CLI 使用

识别并计算单张图片：

```bash
python3 run.py data/samples/printed_basic/printed_basic_001.png --recognizer printed --backend arithmetic
```

自动识别：

```bash
python3 run.py data/samples/printed_2d_layout/printed_2d_001.png --recognizer auto
```

输出调试文件：

```bash
python3 run.py data/samples/printed_basic/printed_basic_001.png --recognizer printed --debug-dir debug/demo
```

可选识别器：

```text
printed | handwritten | calculus | auto
```

可选求解后端：

```text
auto | arithmetic | symbolic
```

## Web UI

```bash
python3 ui/app.py
```

启动后访问 `http://127.0.0.1:7861`。

界面支持图片上传、识别模式切换、计算后端切换、结果展示、token 表格、二值化图和分割框可视化。Auto 模式还会展示路由特征和候选分数。

## 测试

```bash
python3 -m unittest discover -s tests
```

当前本地验证结果：

```text
Ran 76 tests
OK (skipped=5)
```

## 评估

评估脚本默认把 CSV 写入 `reports/evaluation/`，该目录中的 CSV 属于生成物，不提交到版本库。

```bash
python3 tools/evaluate_samples.py --category all
python3 tools/evaluate_samples.py --category auto --confusion-matrix
python3 tools/evaluate_samples.py --category printed_basic --augment-morphology
```

如需指定输出：

```bash
python3 tools/evaluate_samples.py --category auto --output reports/evaluation/auto.csv --confusion-matrix
```

## 项目结构

```text
.
├── data/                 # 样例图片、标签和模板
├── docs/                 # 架构、数据、评估和设计文档
├── reports/evaluation/   # 本地生成的评估 CSV，默认忽略
├── src/                  # 识别、表达式和求解器实现
├── tests/                # 单元测试和回归测试
├── tools/                # 评估和模板生成工具
├── ui/                   # Gradio Web UI
├── requirements.txt
└── run.py
```

## 约束

- 识别阶段只使用传统图像处理和规则方法。
- 计算阶段可以使用本地求解器或符号计算库，但应与识别职责分离。
- 课程展示重点应放在从图片到表达式的转换过程：预处理、分割、模板匹配、结构分析和规则判断。
