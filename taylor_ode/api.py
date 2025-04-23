import numpy as np
from .base import CompositeODESolver
from .solvers import TaylorSolver, RK45Solver, RadauSolver, HybridTaylorSolver
from .utils import analyze_ode

def solve_ode(f, t_span, y0, method='auto', tol=1e-6, **kwargs):
    """
    统一的ODE求解接口
    
    参数:
        f: ODE右侧函数 f(t, y)
        t_span: 时间区间 [t0, tf]
        y0: 初始值
        method: 求解方法 ('auto', 'taylor', 'rk45', 'radau', 'hybrid')
        tol: 容许误差
        **kwargs: 额外参数
            - order: 泰勒展开阶数
            - t_eval: 评估点
            - max_step: 最大步长
            - plot: 是否绘制解曲线
            
    返回:
        (t, y): 时间点和对应的解
    """
    order = kwargs.get('order', 5)
    max_step = kwargs.get('max_step', np.inf)
    plot = kwargs.get('plot', False)
    t_eval = kwargs.get('t_eval', None)
    
    # 如果指定使用自动选择
    if method == 'auto':
        features = analyze_ode(f, t_span, y0)
        method = features['recommended_solver']
        print(f"问题分析: {', '.join(f'{k}={v}' for k, v in features.items() if k != 'error')}")
        print(f"自动选择方法: {method}")
    
    # 创建求解器
    if method == 'taylor':
        solver = TaylorSolver(f, order=order)
    elif method == 'rk45':
        solver = RK45Solver(f)
    elif method == 'radau' or method == 'implicit':
        solver = RadauSolver(f)
    elif method == 'hybrid':
        solver = HybridTaylorSolver(f, order=order)
    else:
        # 创建组合求解器，包含所有方法
        solver = CompositeODESolver(f)
        solver.add_solver(TaylorSolver(f, order=order))
        solver.add_solver(RK45Solver(f))
        solver.add_solver(RadauSolver(f))
        solver.add_solver(HybridTaylorSolver(f, order=order))
    
    # 求解ODE
    t, y = solver.solve(t_span, y0, tol=tol, max_step=max_step, t_eval=t_eval)
    
    # 如果需要绘图
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        
        if np.isscalar(y0):
            plt.plot(t, y, '-')
            plt.xlabel('t')
            plt.ylabel('y')
        else:
            for i in range(len(y0)):
                if i < y.shape[1]:
                    plt.plot(t, y[:, i], '-', label=f'y{i+1}')
            plt.xlabel('t')
            plt.ylabel('y')
            plt.legend()
            
        plt.title(f'ODE Solution using {solver.name}')
        plt.grid(True)
        plt.show()
    
    return t, y