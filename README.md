# firstIntegralsPredictor

## 生成训练集

直接运行 ```datasetGenerator.py```,可以生成名为```dataset.txt```的训练集文件；运行```datasetGenerator_fast.py```可以启动多个线程来进行生成

## 训练模型

生成数据后，直接运行```train.py```可以按照默认值训练出模型

## 测试

在终端运行

```python test_model.py --model_path pth文件路径 --data_path dataset.txt```

可以输入微分方程组（波兰表达式），然后程序会输出波兰表达式形式的首次积分

## 其他程序

```expressionGenerator.py```:随机生成symPy表达式，完全按照论文里的思路

```sympy2Polish.py```,```Polish2Sympy.py```:实现symPy表达式和波兰表达式之间的相互转化

```transformer.py```:直接从Lyapunov论文里复制粘贴过来的transformer架构

这几个文件都在其他文件中被调用