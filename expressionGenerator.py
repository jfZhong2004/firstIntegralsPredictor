import sympy as sp
import random
import numpy as np
from typing import List, Callable, Union

def sub(a, b): return sp.Add(a, sp.Mul(-1, b))
def div(a, b): return sp.Mul(a, sp.Pow(b, -1))

class TreeNode:
    def __init__(self):
        self.type = None  # 'binary'/'unary'/'leaf'
        self.value = None
        self.children = []
        self.parent = None

    def add_child(self, node):
        node.parent = self
        self.children.append(node)

class ExpressionGenerator:
    def __init__(
        self,
        n: int,
        unary_ops: List[Callable],
        binary_ops: List[Callable],
        leaf_vars: List[sp.Symbol],
        leaf_constants: List[Union[int, float]],
        var_prob: float = 0.5
    ):
        self.n = n
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.leaf_vars = leaf_vars
        self.leaf_constants = leaf_constants
        self.var_prob = var_prob
        
        # 动态矩阵配置
        self.base_max_e = 2 * n + 10
        self.base_max_n = n + 10
        self.D = np.zeros((self.base_max_e+2, self.base_max_n+2), dtype=np.int64)
        self._precompute_matrix()

    def _precompute_matrix(self):
        """严格实现附录C的递推公式"""
        # 初始化边界条件
        self.D[0, :] = 0  # D(0, n) = 0 for n > 0
        self.D[:, 0] = 1   # D(e, 0) = 1 for all e
        
        # 填充动态规划矩阵
        for n in range(1, self.base_max_n+1):
            for e in range(1, self.base_max_e+1):
                term1 = self.D[e-1, n] if e-1 >=0 else 0
                term2 = self.D[e+1, n-1] if (e+1 <= self.base_max_e) and (n-1 >=0) else 0
                term3 = self.D[e, n-1] if (e <= self.base_max_e) and (n-1 >=0) else 0
                self.D[e, n] = term1 + term2 + term3

    def _ensure_matrix_capacity(self, required_e, required_n):
        """动态扩展矩阵容量"""
        if required_e > self.base_max_e or required_n > self.base_max_n:
            new_max_e = max(self.base_max_e, required_e + 10)
            new_max_n = max(self.base_max_n, required_n + 10)
            
            new_D = np.zeros((new_max_e+2, new_max_n+2), dtype=np.int64)
            new_D[:self.base_max_e+1, :self.base_max_n+1] = self.D
            
            self.base_max_e, self.base_max_n = new_max_e, new_max_n
            self.D = new_D
            self._precompute_matrix()

    def generate_leaf(self):
        return (random.choice(self.leaf_vars) 
                if random.random() < self.var_prob 
                else random.choice(self.leaf_constants))

    def build_tree_structure(self):
        """完全实现附录C的树生成算法"""
        root = TreeNode()
        empty_nodes = [root]
        remaining_n = self.n
        
        while remaining_n > 0:
            current_e = len(empty_nodes)
            self._ensure_matrix_capacity(current_e, remaining_n)
            
            # 计算所有可能的(k,a)组合及其概率
            valid_combinations = []
            probabilities = []
            for k in range(current_e):
                # 计算unary和binary的概率
                e_after_unary = current_e - k  # unary会消耗1空节点，生成1新节点
                n_after_unary = remaining_n - 1
                prob_unary = self.D[e_after_unary, n_after_unary] if (e_after_unary >=0 and n_after_unary >=0) else 0
                
                e_after_binary = current_e - k + 1  # binary消耗1空节点，生成2新节点
                n_after_binary = remaining_n - 1
                prob_binary = self.D[e_after_binary, n_after_binary] if (e_after_binary <= self.base_max_e and n_after_binary >=0) else 0
                
                if prob_unary + prob_binary > 0:
                    valid_combinations.append((k, 'unary'))
                    probabilities.append(prob_unary)
                    valid_combinations.append((k, 'binary'))
                    probabilities.append(prob_binary)
            
            if not valid_combinations:
                raise ValueError(f"无法在e={current_e}, n={remaining_n}时生成树")
            
            # 归一化概率
            total = sum(probabilities)
            if total == 0:
                raise ValueError("所有组合概率为0")
            probabilities = [p/total for p in probabilities]
            
            # 随机选择(k,a)
            selected_idx = np.random.choice(len(valid_combinations), p=probabilities)
            k, a = valid_combinations[selected_idx]
            
            # 标记前k个空节点为叶子
            for node in empty_nodes[:k]:
                node.type = 'leaf'
            
            # 处理操作符节点
            op_node = empty_nodes[k]
            if a == 'unary':
                op_node.type = 'unary'
                # 添加1个子节点
                child = TreeNode()
                op_node.add_child(child)
                # 更新空节点：k+1后的节点 + 新节点
                empty_nodes = empty_nodes[k+1:] + [child]
                remaining_n -= 1
            elif a == 'binary':
                op_node.type = 'binary'
                # 添加两个子节点
                left = TreeNode()
                right = TreeNode()
                op_node.add_child(left)
                op_node.add_child(right)
                # 更新空节点：k+1后的节点 + 新节点
                empty_nodes = empty_nodes[k+1:] + [left, right]
                remaining_n -= 1
        
        # 标记剩余空节点为叶子
        for node in empty_nodes:
            node.type = 'leaf'
        return root

    def fill_tree(self, node):
        """填充树节点并生成SymPy表达式"""
        if node.type == 'leaf':
            return self.generate_leaf()
        
        if node.type == 'unary':
            op = random.choice(self.unary_ops)
            if len(node.children) != 1:
                raise ValueError("单目操作符应有1个子节点")
            child_expr = self.fill_tree(node.children[0])
            return op(child_expr)
        
        if node.type == 'binary':
            op = random.choice(self.binary_ops)
            if len(node.children) != 2:
                raise ValueError("双目操作符应有2个子节点")
            left = self.fill_tree(node.children[0])
            right = self.fill_tree(node.children[1])
            if op == sub:
                return left - right
            elif op == div:
                return left / right
            else:
                return op(left, right)
        
        raise ValueError("未知节点类型")

    def count_internal_nodes(self, node):
        """统计树结构中的内部节点数"""
        if node.type == 'leaf':
            return 0
        count = 1
        for child in node.children:
            count += self.count_internal_nodes(child)
        return count

    def generate(self):
        if self.n == 0:
            return self.generate_leaf()
        
        structure = self.build_tree_structure()
        actual_nodes = self.count_internal_nodes(structure)
        if actual_nodes != self.n:
            raise RuntimeError(f"节点数不匹配: 预期{self.n}, 实际{actual_nodes}")
        
        return self.fill_tree(structure)

if __name__ == "__main__":
    x = sp.Symbol('x')
    config = {
        'n': 3,
        'unary_ops': [sp.sin, sp.cos, sp.tan],
        'binary_ops': [sp.Add, sp.Mul, sub, div],
        'leaf_vars': [x],
        'leaf_constants': list(range(-5,6)),
        'var_prob': 0.7
    }
    
    try:
        gen = ExpressionGenerator(**config)
        expr = gen.generate()
        print(f"成功生成表达式: {expr}")
    except Exception as e:
        print(f"生成失败: {str(e)}")