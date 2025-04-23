import numpy as np
import warnings

from sympy import factorial

def analyze_ode(f, t_span, y0, samples=10):
    """
    分析ODE问题的特性，帮助选择合适的求解方法
    
    参数:
        f: ODE右侧函数 f(t, y)
        t_span: 时间区间 [t0, tf]
        y0: 初始值
        samples: 采样点数量
        
    返回:
        特性字典
    """
    features = {}
    
    # 采样时间点
    t_points = np.linspace(t_span[0], t_span[1], samples)
    
    # 测试函数行为
    y = y0
    dt = (t_span[1] - t_span[0]) / (samples - 1)
    f_values = []
    is_scalar = np.isscalar(y)
    
    try:
        for t in t_points:
            f_val = f(t, y)
            f_values.append(f_val)
            
            # 简单的欧拉步进用于分析
            if is_scalar:
                y = y + dt * f_val
            else:
                y = y + dt * np.array(f_val)
                
        # 分析f值的变化
        if is_scalar:
            f_changes = [abs(f_values[i+1] - f_values[i]) for i in range(len(f_values)-1)]
            f_max = max(abs(f) for f in f_values)
            f_min = min(abs(f) for f in f_values)
        else:
            f_changes = [np.linalg.norm(np.array(f_values[i+1]) - np.array(f_values[i])) 
                         for i in range(len(f_values)-1)]
            f_max = max(np.linalg.norm(np.array(f)) for f in f_values)
            f_min = min(np.linalg.norm(np.array(f)) for f in f_values) 
        
        # 检测刚性
        features['stiffness_ratio'] = f_max / (f_min + 1e-10)
        features['is_stiff'] = features['stiffness_ratio'] > 1000
        
        # 检测振荡性
        if is_scalar:
            sign_changes = sum(1 for i in range(len(f_values)-1) 
                              if np.sign(f_values[i]) != np.sign(f_values[i+1]))
            features['is_oscillatory'] = sign_changes > 0
        else:
            # 对于系统，检查第一个分量的符号变化
            sign_changes = sum(1 for i in range(len(f_values)-1) 
                              if np.sign(f_values[i][0]) != np.sign(f_values[i+1][0]))
            features['is_oscillatory'] = sign_changes > 0
        
        # 检测非线性
        features['max_rate_of_change'] = max(f_changes) if f_changes else 0
        features['is_highly_nonlinear'] = features['max_rate_of_change'] > 10 * f_max
        
        # 推荐求解器
        if features['is_stiff']:
            features['recommended_solver'] = 'implicit'
        elif features['is_highly_nonlinear']:
            features['recommended_solver'] = 'rk'
        elif features['is_oscillatory']:
            features['recommended_solver'] = 'hybrid'
        else:
            features['recommended_solver'] = 'taylor'
            
    except Exception as e:
        warnings.warn(f"ODE分析过程出错: {e}")
        features['error'] = str(e)
        features['recommended_solver'] = 'rk'  # 出错时使用最可靠的方法
    
    return features

def estimate_optimal_order(f, t, y, max_order=10):
    """
    估计泰勒展开的最佳阶数
    
    参数:
        f: ODE右侧函数
        t: 当前时间点
        y: 当前值
        max_order: 最大考虑阶数
        
    返回:
        推荐阶数
    """
    derivatives = []
    
    try:
        # 计算0阶导数
        derivatives.append(y)
        
        # 计算1阶导数
        derivatives.append(f(t, y))
        
        # 使用有限差分近似计算高阶导数
        for k in range(2, max_order + 1):
            if k == 2:
                # 二阶导数特殊处理
                h = 1e-6
                f1 = f(t, y)
                y2 = y + h * f1
                f2 = f(t + h, y2)
                deriv = (f2 - f1) / h
            else:
                # 高阶导数使用近似
                h = 1e-6
                prev_deriv = derivatives[k-1]
                y2 = y + h * derivatives[1]  # 使用一阶导数前进
                next_deriv = (f(t + h, y2) - derivatives[1]) / h  # 近似高阶导数
                
                # 应用衰减避免数值不稳定性
                deriv = next_deriv * (0.1 ** (k-2))
                
            derivatives.append(deriv)
        
        # 分析导数序列的收敛性
        normalized_derivs = [abs(derivatives[k]) / (factorial(k) + 1e-10) for k in range(1, len(derivatives))]
        
        # 找到首次满足收敛条件的阶数
        for k in range(len(normalized_derivs) - 1):
            if normalized_derivs[k+1] < 0.1 * normalized_derivs[k]:
                return k + 1  # 加1是因为k从0开始而阶数从1开始
                
        # 如果没有明显收敛，返回一个中等阶数
        return min(5, max_order)
        
    except Exception as e:
        warnings.warn(f"阶数估计失败: {e}")
        return 4  # 出错时返回安全的中等阶数