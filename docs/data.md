# 数据集说明

样例图片位于 `data/samples/`，标签位于 `data/labels/`。标签文件与图片同名，只是扩展名为 `.txt`。

示例：

```text
data/samples/printed_basic/printed_basic_001.png
data/labels/printed_basic/printed_basic_001.txt
```

## 类别

| 类别 | 图片数量 | 标签数量 | 说明 |
| --- | ---: | ---: | --- |
| `printed_basic` | 20 | 20 | 基础印刷体四则运算。 |
| `printed_decimal_negative` | 20 | 20 | 负数、小数、乘除显示规范样例。 |
| `printed_2d_layout` | 20 | 20 | 分数、上下标、根号等二维结构样例。 |
| `handwritten_basic` | 105 | 105 | 21 类基础手写字符，每类 5 张。 |
| `calculus` | 10 | 10 | 导数、积分、极限样例。 |
| `test` | 1 | 1 | 手写诊断样例。 |

## 模板

印刷体模板位于 `data/templates/printed_basic/`。模板生成工具：

```bash
python3 tools/generate_printed_templates.py
```

## 用途

这些数据用于：

- 单元测试和回归测试。
- 分类评估。
- Auto router 路由验证。
- 课程展示中的可复现实验样例。
