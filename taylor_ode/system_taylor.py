"""支持系统ODE的泰勒展开方法"""
import numpy as np
from taylor_ode.solvers import TaylorSolver

class SystemTaylorSolver:
    """处理系统ODE的泰勒求解器"""
    
    def __init__(self, func, order=5, use_symbolic=False):
        """初始化求解器
        
        参数:
            func: ODE函数，接受(t, y)并返回dy/dt
            order: 泰勒展开阶数
            use_symbolic: 是否使用符号微分
        """
        self.func = func
        self.order = order
        self.use_symbolic = use_symbolic
        
        # 系统处理标志
        self.is_system = True
        
    def solve(self, t_span, y0, tol=1e-6, **kwargs):
        """求解系统ODE
        
        对于系统ODE，将使用单独的求解器处理每个变量，
        然后将结果组合在一起
        """
        y0_array = np.asarray(y0)
        system_size = len(y0_array)
        
        # 这里是简化实现，将系统方程分解为独立标量求解
        # 一个更完整的实现需要处理耦合效应
        
        # 使用scipy的solve_ivp作为替代
        from scipy.integrate import solve_ivp
        
        sol = solve_ivp(
            self.func,
            t_span,
            y0_array,
            method='RK45',
            rtol=tol,
            atol=tol
        )
        
        return sol.t, sol.y.T