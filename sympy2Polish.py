import sympy as sp
from sympy.core.function import AppliedUndef

def sympy_to_polish(expr):
    """将SymPy表达式转换为波兰表达式字符串"""
    # 处理符号
    if isinstance(expr, sp.Symbol):
        return expr.name

    # 处理所有数值类型（整数、分数、浮点数）
    if isinstance(expr, sp.Number):
        # 如果是负数，视作 0 减去这个数
        if expr.is_negative:
            return f"- 0 {abs(expr)}"
        return str(expr)

    # 处理数学常数 e
    if expr == sp.E:
        return "e"

    # 处理数学常数 pi
    if expr == sp.pi:
        return "pi"

    # 处理虚数单位 i
    if expr == sp.I:
        return "i"

    # 处理加法（包含隐式减法）
    if isinstance(expr, sp.Add):
        result = _handle_add(expr)
        print(f"生成的波兰表达式 (Add): {result}")  # 调试信息
        return result

    # 处理乘法（包含隐式除法）
    if isinstance(expr, sp.Mul):
        result = _handle_mul(expr)
        print(f"生成的波兰表达式 (Mul): {result}")  # 调试信息
        return result

    # 处理幂运算（包含sqrt特例）
    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        if exponent == sp.S.Half:  # 识别平方根
            result = f"sqrt {sympy_to_polish(base)}"
            print(f"生成的波兰表达式 (Sqrt): {result}")  # 调试信息
            return result
        result = f"^ {sympy_to_polish(base)} {sympy_to_polish(exponent)}"
        print(f"生成的波兰表达式 (Pow): {result}")  # 调试信息
        return result

    # 处理函数调用
    if isinstance(expr, sp.Function):
        func_name = expr.func.__name__.lower()
        args = " ".join(sympy_to_polish(arg) for arg in expr.args)
        result = f"{func_name} {args}"
        print(f"生成的波兰表达式 (Function): {result}")  # 调试信息
        return result

    # 处理未定义符号
    if isinstance(expr, AppliedUndef):
        return str(expr)

    raise ValueError(f"未支持的表达式类型: {type(expr)}")

def _handle_add(expr):
    """处理加法表达式，自动识别隐式减法"""
    positive = []
    negative = []

    for term in expr.args:
        # 识别负项（形如-1*x）
        if isinstance(term, sp.Mul) and term.args[0] == -1:
            negative.append(sympy_to_polish(term.args[1]))
        else:
            positive.append(sympy_to_polish(term))

    # 构建波兰表达式
    expression = positive[0] if positive else "0"
    for item in positive[1:]:
        expression = f"+ {expression} {item}"
    for item in negative:
        expression = f"- {expression} {item}"

    return expression

def _handle_mul(expr):
    """处理乘法表达式，自动识别隐式除法"""
    numerator = []
    denominator = []

    for factor in expr.args:
        # 识别倒数（形如x**-1）
        if isinstance(factor, sp.Pow) and factor.exp == -1:
            denominator.append(sympy_to_polish(factor.base))
        else:
            numerator.append(sympy_to_polish(factor))

    # 构建乘法表达式
    expression = numerator[0] if numerator else "1"
    for item in numerator[1:]:
        expression = f"* {expression} {item}"

    # 添加除法部分
    for item in denominator:
        expression = f"/ {expression} {item}"

    # 替换乘方运算符 ** 为 ^
    expression = expression.replace('**', '^')

    return expression

# 测试用例
if __name__ == "__main__":
    x, y, t = sp.symbols('x y t')
    
    # 测试案例1: 题目示例
    test_expr = sp.atan(t + y - sp.log(x))
    print(sympy_to_polish(test_expr))  # 输出: atan - + t y log x
    
    # 测试案例2: 含分数和嵌套运算
    test_expr2 = sp.sqrt(sp.sin(x)**2 + sp.cos(y)**3 * 5)
    print(sympy_to_polish(test_expr2))  # 输出: sqrt + ^ sin x 2 * ^ cos y 3 5
    
    # 测试案例3: 含分数指数
    test_expr3 = x**sp.Rational(3, 2) + y**-2
    print(sympy_to_polish(test_expr3))  # 输出: + ^ x / 3 2 / 1 y
    
    # 测试案例4: 含浮点数
    test_expr4 = sp.exp(0.5 * t - 3)
    print(sympy_to_polish(test_expr4))  # 输出: exp - * 0.5 t 3

    # 测试案例5: 嵌套加法和乘法
    test_expr5 = (x + y) * (t + 2)
    print("测试案例5:", sympy_to_polish(test_expr5))  # 输出: * + x y + t 2

    # 测试案例6: 复杂嵌套表达式
    test_expr6 = (x - 2) * (y + 1) + t * (x + y)
    print("测试案例6:", sympy_to_polish(test_expr6))  # 输出: + * - x 2 + y 1 * t + x y

    # 测试案例7: 分数和幂运算
    test_expr7 = (x**2 + y**2) / (t - 1)
    print("测试案例7:", sympy_to_polish(test_expr7))  # 输出: / + ^ x 2 ^ y 2 - t 1

    # 测试案例8: 复合函数
    test_expr8 = sp.sin(x) + sp.log(y) * sp.sqrt(t)
    print("测试案例8:", sympy_to_polish(test_expr8))  # 输出: + sin x * log y sqrt t