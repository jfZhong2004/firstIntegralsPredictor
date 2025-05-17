import sympy as sp

def polish_to_sympy(expr):
    """将波兰表达式字符串转换为 SymPy 表达式"""
    print(f"解析波兰表达式: {expr}")  # 调试信息
    tokens = expr.split()
    stack = []

    while tokens:
        token = tokens.pop()
        print(f"当前 token: {token}, 当前栈: {stack}, 栈元素类型: {[type(item) for item in stack]}")  # 调试信息
        try:
            if token in {"+", "-", "*", "/", "^"}:
                # 操作符处理
                if len(stack) < 2:
                    raise ValueError("波兰表达式无效: 操作数不足")
                b = stack.pop()
                a = stack.pop()
                # 统一处理所有类型的操作数
                try:
                    if token == "*":
                        stack.append(sp.Mul(b, a))
                    elif token == "+":
                        stack.append(sp.Add(b, a))
                    elif token == "-":
                        stack.append(sp.Add(b, -a))
                    elif token == "/":
                        stack.append(sp.Mul(b, sp.Pow(a, -1)))
                    elif token == "^":
                        stack.append(sp.Pow(b, a))
                except Exception as e:
                    raise TypeError(f"不支持的操作数类型: {type(b)} 和 {type(a)}，错误: {e}")
            elif token == "sqrt":
                # 特殊函数 sqrt
                if not stack:
                    raise ValueError("波兰表达式无效: 操作数不足")
                a = stack.pop()
                stack.append(sp.sqrt(a))
            elif token == "log":
                # 特殊函数 log
                if not stack:
                    raise ValueError("波兰表达式无效: 操作数不足")
                a = stack.pop()
                stack.append(sp.log(a))
            elif token == "sin":
                # 特殊函数 sin
                if not stack:
                    raise ValueError("波兰表达式无效: 操作数不足")
                a = stack.pop()
                stack.append(sp.sin(a))
            elif token == "cos":
                # 特殊函数 cos
                if not stack:
                    raise ValueError("波兰表达式无效: 操作数不足")
                a = stack.pop()
                stack.append(sp.cos(a))
            elif token == "tan":
                # 特殊函数 tan
                if not stack:
                    raise ValueError("波兰表达式无效: 操作数不足")
                a = stack.pop()
                stack.append(sp.tan(a))
            elif token == "exp":
                # 特殊函数 exp
                if not stack:
                    raise ValueError("波兰表达式无效: 操作数不足")
                a = stack.pop()
                stack.append(sp.exp(a))
            elif token == "atan":
                # 特殊函数 atan
                if not stack:
                    raise ValueError("波兰表达式无效: 操作数不足")
                a = stack.pop()
                stack.append(sp.atan(a))
            elif token == "pi":
                # 常量 pi
                stack.append(sp.pi)
            elif token == "e":
                # 常量 e
                stack.append(sp.E)
            elif token == "i":
                # 虚数单位 i
                stack.append(sp.I)
            else:
                # 处理数字和符号
                try:
                    stack.append(sp.sympify(token))
                except sp.SympifyError:
                    stack.append(sp.Symbol(token))
        except IndexError:
            raise ValueError("波兰表达式无效: 栈操作失败")

    print(f"最终栈: {stack}")  # 调试信息
    if len(stack) != 1:
        raise ValueError("波兰表达式无效: 栈中剩余多余元素")
    return stack[0]

# 测试用例
if __name__ == "__main__":
    # 示例波兰表达式
    polish_expr1 = "atan - + t y log x"
    polish_expr2 = "sqrt + ^ sin x 2 * ^ cos y 3 5"
    polish_expr3 = "+ ^ x / 3 2 / 1 y"
    polish_expr4 = "exp - * 0.5 t 3"

    # 转换为 SymPy 表达式
    sympy_expr1 = polish_to_sympy(polish_expr1)
    sympy_expr2 = polish_to_sympy(polish_expr2)
    sympy_expr3 = polish_to_sympy(polish_expr3)
    sympy_expr4 = polish_to_sympy(polish_expr4)

    # 打印结果
    print("SymPy 表达式 1:", sympy_expr1)
    print("SymPy 表达式 2:", sympy_expr2)
    print("SymPy 表达式 3:", sympy_expr3)
    print("SymPy 表达式 4:", sympy_expr4)