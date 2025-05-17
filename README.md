# firstIntegrals

`firstIntegrals` 是一个用于生成微分方程组及其首次积分的项目。该项目的主要目标是通过随机生成微分方程组的首次积分，反向计算出其对应的微分方程组，生成训练数据集，并使用 Transformer 模型训练一个能够预测微分方程组首次积分的模型。

---

## 功能概述

1. **首次积分生成**：
   - 使用 `ExpressionGenerator` 随机生成符号表达式，作为微分方程组的首次积分。

2. **反向计算微分方程组**：
   - 根据生成的首次积分，反向计算出对应的微分方程组。

3. **数据集生成**：
   - 将首次积分及其对应的微分方程组保存为波兰表达式，存储在 `dataset.txt` 文件中。

4. **Transformer 模型训练**：
   - 使用生成的数据集训练 Transformer 模型，使其能够预测微分方程组的首次积分。

5. **波兰表达式转换**：
   - 提供 `sympy_to_polish` 和 `polish_to_sympy` 函数，用于在 SymPy 表达式和波兰表达式之间进行转换。

---

## 文件结构

- **`main.py`**：
  - 项目的主入口，负责生成首次积分、反向计算微分方程组、验证并保存结果。
  
- **`sympy2Polish.py`**：
  - 提供 `sympy_to_polish` 函数，将 SymPy 表达式转换为波兰表达式。

- **`Polish2Sympy.py`**：
  - 提供 `polish_to_sympy` 函数，将波兰表达式还原为 SymPy 表达式。

- **`expressionGenerator.py`**：
  - 实现 `ExpressionGenerator` 类，用于随机生成符号表达式。

- **`transformer.py`**：
  - 包含 Transformer 模型的实现，用于处理数学表达式的序列建模。

- **`train.py`**：
  - 用于训练 Transformer 模型的脚本。

- **`test_model.py`**：
  - 用于加载训练好的模型并测试其性能。

- **`dataset.txt`**：
  - 保存生成的微分方程组及其首次积分的波兰表达式。

---

## 安装与运行

### 环境依赖

- Python 3.8+
- 必要的 Python 包：
  - `sympy`
  - `numpy`
  - `torch`

可以通过以下命令安装依赖：

```bash
pip install sympy numpy torch
```

### 运行项目

1. **生成数据集**：
   运行 `main.py` 以生成首次积分及其对应的微分方程组，并保存到 `dataset.txt` 文件中。

   ```bash
   python main.py
   ```

2. **训练模型**：
   使用 `train.py` 训练 Transformer 模型。

   ```bash
   python train.py --data_path dataset.txt --save_dir saved_models
   ```

3. **测试模型**：
   使用 `test_model.py` 测试训练好的模型。

   ```bash
   python test_model.py --model_path saved_models/model_epoch_10.pt
   ```

---

## 示例

### 数据集示例

以下是 `dataset.txt` 中的示例内容：

```
(2*x**2 + x*y + 2*x + y)/(t*y - 2*t)
(-4*x**2 - 4*x*y - y**2)/(t*y - 2*t)
t*(2*x + y)
x*(x + y) + y + 1
/ + + + y * 2 x * 2 ^ x 2 * x y + * - 0 2 t * t y , / - + * - 0 4 ^ x 2 * * - 0 4 x y ^ y 2 + * - 0 2 t * t y ; * t + y * 2 x , + + 1 y * x + x y
```

### 波兰表达式转换

- SymPy 表达式：`atan(t + y - log(x))`
- 转换为波兰表达式：`atan - + t y log x`
- 还原为 SymPy 表达式：`atan(t + y - log(x))`

---

## 项目目标

通过生成和反向计算微分方程组及其首次积分，训练一个 Transformer 模型，使其能够预测给定微分方程组的首次积分。

---

## 贡献

如果您发现任何问题或有改进建议，欢迎提交 Issue 或 Pull Request。

---

## 许可证

本项目遵循 MIT 许可证。