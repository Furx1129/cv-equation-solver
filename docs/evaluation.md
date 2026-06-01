# 评估与测试

## 单元测试

```bash
python3 -m unittest discover -s tests
```

当前本地验证结果：

```text
Ran 76 tests
OK (skipped=5)
```

测试覆盖预处理、分割、模板匹配、二维结构、微积分结构、表达式序列化、求解器、auto router、交叉验证和符号消歧等模块。

## 批量评估

评估脚本：

```bash
python3 tools/evaluate_samples.py --category all
```

默认输出：

```text
reports/evaluation/evaluation_results.csv
```

`reports/evaluation/*.csv` 是生成物，默认被 Git 忽略。需要保留某次结果时，建议在报告正文或提交说明中记录命令、代码版本和摘要指标，而不是把整份 CSV 放在项目根目录。

## 常用命令

评估所有分类识别器：

```bash
python3 tools/evaluate_samples.py --category all
```

评估自动路由并输出混淆矩阵：

```bash
python3 tools/evaluate_samples.py --category auto --confusion-matrix
```

指定输出文件：

```bash
python3 tools/evaluate_samples.py --category auto --output reports/evaluation/auto.csv --confusion-matrix
```

测试形态学扰动鲁棒性：

```bash
python3 tools/evaluate_samples.py --category printed_basic --augment-morphology
```

调试失败样例：

```bash
python3 tools/evaluate_samples.py --category auto --debug-failures
```

## 支持的类别

- `all`
- `auto`
- `printed_basic`
- `printed_decimal_negative`
- `printed_2d_layout`
- `handwritten_basic`
- `calculus`

## 输出字段

CSV 包含标签、预测、匹配结果、路由类别、候选分数、布局类型、SymPy 文本、求解器结果、低置信度 token 数、未知 token 数和失败阶段等诊断信息。

使用 `--confusion-matrix` 时，脚本还会在同一目录生成：

```text
<output-stem>.confusion_matrix.csv
```
