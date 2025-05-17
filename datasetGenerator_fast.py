import random
from sympy import symbols, diff, simplify, linsolve, sin, cos, tan, log, exp, asin, acos, atan, sqrt, Add, Mul, Pow
from sympy.abc import x, y, t
from expressionGenerator import ExpressionGenerator
from sympy2Polish import sympy_to_polish
from Polish2Sympy import polish_to_sympy  # 导入 polish_to_sympy 函数
import multiprocessing

# 可用变量列表
variables = [x, y, t]

def generate_integral(generator_config):
    """使用ExpressionGenerator生成首次积分"""
    generator = ExpressionGenerator(**generator_config)
    return generator.generate()

def sub(a, b):
    return Add(a, Mul(-1, b))

def div(a, b):
    return Mul(a, Pow(b, -1))

def worker_process(process_id, generator_config, target_count, length_threshold):
    """每个进程的工作函数"""
    generated_count = 0


    while generated_count < target_count:
        try:
            # 步骤1：生成两个首次积分
            f = generate_integral(generator_config)
            g = generate_integral(generator_config)

            # 步骤2：计算偏导数并建立方程组
            df_dx = diff(f, x)
            df_dy = diff(f, y)
            df_dt = diff(f, t)

            dg_dx = diff(g, x)
            dg_dy = diff(g, y)
            dg_dt = diff(g, t)

            dxdt, dydt = symbols('dxdt dydt')
            eq1 = df_dx * dxdt + df_dy * dydt + df_dt
            eq2 = dg_dx * dxdt + dg_dy * dydt + dg_dt

            # 解线性方程组
            solution = linsolve([eq1, eq2], (dxdt, dydt))

            if not solution or len(solution) > 1:
                continue

            dxdt_expr, dydt_expr = list(solution)[0]

            # 检查解中是否包含自引用
            if dxdt in dxdt_expr.free_symbols or dydt in dxdt_expr.free_symbols or \
               dxdt in dydt_expr.free_symbols or dydt in dydt_expr.free_symbols:
                continue

            # 检查退化条件
            dxdt_vars = dxdt_expr.free_symbols
            dydt_vars = dydt_expr.free_symbols

            if dxdt_vars.issubset({x, t}) or dydt_vars.issubset({y, t}):
                continue

            # 步骤3：验证解的正确性
            substituted_f = df_dx * dxdt_expr + df_dy * dydt_expr + df_dt
            substituted_g = dg_dx * dxdt_expr + dg_dy * dydt_expr + dg_dt

            simplified_f = simplify(substituted_f)
            simplified_g = simplify(substituted_g)

            # 符号验证
            valid_f = simplified_f == 0
            valid_g = simplified_g == 0

            if valid_f and valid_g:
                dxdt_polish = sympy_to_polish(dxdt_expr)
                dydt_polish = sympy_to_polish(dydt_expr)
                f_polish = sympy_to_polish(f)
                g_polish = sympy_to_polish(g)

                trainingData = f"{dxdt_polish} , {dydt_polish} ; {f_polish} , {g_polish}"

                # 检查长度阈值
                if len(trainingData) > length_threshold:
                    continue

                # 写入数据集文件
                with open(f'dataset_{process_id}.txt', 'a') as f:
                    f.write(trainingData + '\n')

                generated_count += 1

                # 每生成10个输出提示语
                if generated_count % 10 == 0:
                    print(f"进程 {process_id}: 已生成 {generated_count} 个微分方程组-首次积分对")

        except Exception as e:
            # 捕获异常并继续
            print(f"进程 {process_id}: 发生错误 - {e}")
            continue

    print(f"进程 {process_id}: 目标生成数量 {target_count} 已完成")

def main():
    # 配置ExpressionGenerator参数
    generator_config = {
        'n': 4,  # 内部节点数量
        'unary_ops': [sin, cos, tan, log, exp, asin, acos, atan, sqrt],  # 一元运算符
        'binary_ops': [Add, Mul, sub, div],  # 二元运算符
        'leaf_vars': variables,  # 变量
        'leaf_constants': [-2, -1, 1, 2],  # 常量
        'var_prob': 0.7  # 选择变量的概率
    }

    target_count = 30  # 每个进程的目标生成数量
    length_threshold = 800  # trainingData长度阈值
    num_processes = multiprocessing.cpu_count()  # 使用 CPU 核心数

    # 启动多个进程
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=worker_process, args=(i, generator_config, target_count, length_threshold))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print("所有进程已完成")

if __name__ == "__main__":
    main()