"""专门处理系统ODE的模块"""
import numpy as np
from scipy.integrate import solve_ivp

def solve_system_ode(func, t_span, y0, tol=1e-6, method='RK45'):
    """专门处理系统ODE的求解函数
    
    参数:
        func: 系统ODE函数，接受(t, y)并返回dy/dt
        t_span: [t_start, t_end]时间范围
        y0: 初始状态向量
        tol: 误差容限
        method: 求解方法，默认RK45
        
    返回:
        t: 时间点数组
        y: 对应的状态值数组，形状为(len(t), len(y0))
    """
    # 将y0标准化为numpy数组
    y0_array = np.asarray(y0)
    
    # 确保y0是向量形式
    if y0_array.ndim == 0:
        y0_array = np.array([y0_array])
    
    # 包装func以确保它能处理向量输入
    def wrapped_func(t, y):
        try:
            return np.asarray(func(t, y))
        except Exception as e:
            # 尝试单独计算每个元素
            if len(y) > 1:
                return np.array([func(t, yi) for yi in y])
            raise e
    
    # 使用solve_ivp求解
    print(f"系统ODE求解: 使用{method}方法，容限={tol}")
    sol = solve_ivp(
        wrapped_func, 
        t_span, 
        y0_array, 
        method=method, 
        rtol=tol, 
        atol=tol,
        dense_output=True
    )
    
    # 返回结果
    return sol.t, sol.y.T

# 检测函数是否是系统ODE
def is_system_ode(func, y0):
    """检测给定的函数是否为系统ODE
    
    参数:
        func: ODE函数
        y0: 初始值
        
    返回:
        bool: 是否是系统ODE
    """
    # 如果y0是向量，很可能是系统ODE
    if hasattr(y0, '__len__') and len(y0) > 1:
        return True
    
    # 尝试计算一个值
    try:
        result = func(0, y0)
        # 如果结果是向量，则是系统ODE
        if hasattr(result, '__len__') and len(result) > 1:
            return True
    except:
        pass
        
    return False