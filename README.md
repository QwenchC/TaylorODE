# TaylorODE - 基于泰勒展开的微分方程高精度解法（对应毕业设计论文《泰勒公式及其在近似计算中的应用拓展》的第5章）

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-orange.svg)](https://scipy.org/)

TaylorODE 是一个高精度微分方程数值解法库，基于泰勒展开和自适应步长控制，适用于需要高精度局部解的科学计算场景。

## 核心特性

- **高阶泰勒展开**：支持可配置阶数的泰勒展开，提供高精度局部解
- **自适应步长控制**：基于截断误差的智能步长调整，平衡精度与效率
- **性能优化**：提供多种性能优化选项，支持JIT编译和GPU加速
- **系统ODE支持**：原生支持ODE系统的高精度求解

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 简单ODE求解示例

```python
import numpy as np
from taylor_ode import TaylorODESolver

# 定义ODE函数: dy/dt = -0.5*y
def exponential_decay(t, y):
    return -0.5 * y

# 创建求解器 (默认5阶泰勒展开)
solver = TaylorODESolver(exponential_decay)

# 求解ODE
t_span = [0, 10]
y0 = 1.0
t, y = solver.solve(t_span, y0, tol=1e-6)

# 计算精确解进行比较
exact = y0 * np.exp(-0.5 * t)
error = np.max(np.abs(y - exact))
print(f"最大误差: {error:.2e}")
```

### 系统ODE求解示例

```python
from taylor_ode import TaylorSystemSolver

# 定义Lotka-Volterra系统
def lotka_volterra(t, state):
    x, y = state
    dx_dt = 1.5*x - x*y
    dy_dt = x*y - 3.0*y
    return np.array([dx_dt, dy_dt])

# 创建系统求解器
solver = TaylorSystemSolver(lotka_volterra, order=8)

# 求解系统
t_span = [0, 15]
initial_state = [1.0, 1.0]  # [猎物, 捕食者]
t, states = solver.solve(t_span, initial_state, tol=1e-6)

# states数组包含所有时间点的解
```

## 性能对比

TaylorODE在高精度要求下的性能对比：

| 方法 | 1e-3误差 | 1e-6误差 | 1e-9误差 |
|------|----------|----------|----------|
| TaylorODE | 快 | 非常快 | 极快 |
| RK45 | 快 | 慢 | 非常慢 |

## 应用场景

- **航天轨道计算**：卫星和航天器轨道预测
- **量子力学模拟**：需要高精度的薛定谔方程数值解
- **化学反应动力学**：刚性微分方程系统求解
- **电路仿真**：高精度瞬态分析

## 项目结构

```
TaylorODE/
├── taylor_ode/         # 核心代码
│   ├── core.py         # 核心解算法
│   ├── optimizations.py # 性能优化
├── examples/           # 示例代码
│   ├── simple_ode.py   # 简单ODE示例
│   ├── orbital.py      # 卫星轨道预测
│   └── compare.py      # 与传统方法比较
└── tests/              # 测试代码
```

## 运行

```
# 基础 ODE 求解/泰勒展开与传统方法性能对比/高级用法模式/泰勒展开评估模式/卫星轨道预测演示
python examples/simple_ode.py
python examples/compare.py
python examples/advanced_usage.py
python examples/evaluate_taylor.py
python examples/orbital.py
```

## 许可证

MIT
