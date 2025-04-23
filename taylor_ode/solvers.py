import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import factorial
from .base import ODESolverBase

class TaylorSolver(ODESolverBase):
    """重构的泰勒展开ODE求解器"""
    
    def __init__(self, f, order=5):
        """初始化求解器"""
        self.f = f
        self.order = order
        from .core import TaylorODESolver
        self._solver = TaylorODESolver(f, order=order)
    
    def solve(self, t_span, y0, tol=1e-6, **kwargs):
        """求解ODE"""
        # 过滤出TaylorODESolver支持的参数
        supported_kwargs = {
            'max_step': kwargs.get('max_step', np.inf),
            'min_step': kwargs.get('min_step', 1e-8)
        }
        
        # 如果提供了t_eval，使用fixed_points参数
        if 't_eval' in kwargs:
            supported_kwargs['fixed_points'] = kwargs['t_eval']
            
        return self._solver.solve(t_span, y0, tol=tol, **supported_kwargs)
    
    def step(self, t, y, h):
        """执行单步求解"""
        return self._solver.taylor_step(t, y, h)
    
    @property
    def name(self):
        return f"Taylor(order={self.order})"

class RK45Solver(ODESolverBase):
    """Runge-Kutta 4-5阶求解器封装"""
    
    def __init__(self, f):
        """初始化求解器"""
        self.f = f
    
    def solve(self, t_span, y0, tol=1e-6, **kwargs):
        """求解ODE"""
        t_eval = kwargs.get('t_eval', None)
        dense_output = kwargs.get('dense_output', False)
        
        result = solve_ivp(
            self.f, t_span, [y0] if np.isscalar(y0) else y0, 
            method='RK45', rtol=tol, atol=tol/10,
            t_eval=t_eval, dense_output=dense_output
        )
        
        if np.isscalar(y0):
            return result.t, result.y[0]
        else:
            return result.t, result.y.T
    
    def step(self, t, y, h):
        """执行单步求解 - 对于RK45使用简化实现"""
        k1 = self.f(t, y)
        k2 = self.f(t + h/2, y + h*k1/2)
        k3 = self.f(t + h/2, y + h*k2/2)
        k4 = self.f(t + h, y + h*k3)
        
        y_next = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # 简单的误差估计
        error = h * abs(k1 - k4) / 6
        
        return y_next, error
    
    @property
    def name(self):
        return "RK45"

class RadauSolver(ODESolverBase):
    """Radau隐式求解器封装 - 适合刚性问题"""
    
    def __init__(self, f):
        """初始化求解器"""
        self.f = f
    
    def solve(self, t_span, y0, tol=1e-6, **kwargs):
        """求解ODE"""
        t_eval = kwargs.get('t_eval', None)
        
        result = solve_ivp(
            self.f, t_span, [y0] if np.isscalar(y0) else y0, 
            method='Radau', rtol=tol, atol=tol/10,
            t_eval=t_eval
        )
        
        if np.isscalar(y0):
            return result.t, result.y[0]
        else:
            return result.t, result.y.T
    
    def step(self, t, y, h):
        """对于隐式方法，单步实现较复杂，这里提供简化版本"""
        # 为保证接口一致，使用求解代替，但这样效率较低
        result = solve_ivp(
            self.f, [t, t+h], [y] if np.isscalar(y) else y, 
            method='Radau', rtol=1e-6, atol=1e-7
        )
        
        last_idx = -1
        y_next = result.y[0, last_idx] if np.isscalar(y) else result.y[:, last_idx]
        
        # 简单的误差估计
        error = 1e-6 * abs(y_next)
        
        return y_next, error
    
    @property
    def name(self):
        return "Radau(Implicit)"

class HybridTaylorSolver(ODESolverBase):
    """混合泰勒展开求解器 - 结合泰勒展开和传统方法的优点"""
    
    def __init__(self, f, order=5, fallback_method='RK45'):
        """初始化混合求解器"""
        self.f = f
        self.order = order
        self.fallback_method = fallback_method
        self._solver_name = f"hybrid"  # 改成_solver_name避免与属性方法同名
        
        # 初始化求解器
        from .core import TaylorODESolver
        self.taylor_solver = TaylorODESolver(f, order=order)
        
        # 初始化备选求解器
        if fallback_method == 'RK45':
            self.fallback_solver = RK45Solver(f)
        elif fallback_method == 'Radau':
            self.fallback_solver = RadauSolver(f)
        else:
            raise ValueError(f"不支持的备选方法: {fallback_method}")
    
    # 添加缺失的name属性方法
    @property
    def name(self):
        """返回求解器名称"""
        return f"Hybrid(Taylor{self.order}/{self.fallback_method})"
    
    def solve(self, t_span, y0, tol=1e-6, **kwargs):
        """增强的solve方法确保总是返回有效结果"""
        # 检查是否是系统ODE
        is_vector = hasattr(y0, '__len__') and len(y0) > 1
        
        # 对于系统ODE，直接使用fallback_solver
        if is_vector:
            print(f"系统ODE检测：优先使用{self.fallback_method}求解")
            try:
                return self.fallback_solver.solve(t_span, y0, tol=tol, **kwargs)
            except Exception as e:
                print(f"备选求解器也失败: {e}")
                # 出错时使用scipy的solve_ivp
                from scipy.integrate import solve_ivp
                sol = solve_ivp(self.f, t_span, y0, method='RK45', rtol=tol, atol=tol)
                return sol.t, sol.y.T
        
        # 对于标量ODE，使用混合策略
        try:
            # 步长限制避免过小步长导致的问题
            kwargs['min_step'] = kwargs.get('min_step', 1e-10)
            
            # 使用原始策略求解
            t, y = [], []
            t_current = t_span[0]
            y_current = y0
            
            while t_current < t_span[1]:
                # 判断是使用泰勒步长还是其他方法
                if self._should_use_taylor_step(t_current, y_current, kwargs.get('max_step', 0.1)):
                    # 使用泰勒方法
                    y_next, error, h_used = self.taylor_solver.step(t_current, y_current, 
                                                                   kwargs.get('max_step', 0.1))
                    # 更新
                    t.append(t_current)
                    y.append(y_current)
                    t_current += h_used
                    y_current = y_next
                else:
                    # 使用备选方法计算剩余部分
                    remaining_span = [t_current, t_span[1]]
                    t_fallback, y_fallback = self.fallback_solver.solve(remaining_span, y_current, tol=tol)
                    
                    # 合并结果
                    if len(t) > 0:
                        t = np.concatenate([t, t_fallback[1:]])
                        y = np.concatenate([y, y_fallback[1:]])
                    else:
                        t = t_fallback
                        y = y_fallback
                    break
                    
            # 确保最后一点是终点
            if t[-1] < t_span[1]:
                # 计算终点值
                remaining_span = [t[-1], t_span[1]]
                t_end, y_end = self.fallback_solver.solve(remaining_span, y[-1], tol=tol)
                t = np.concatenate([t, [t_span[1]]])
                y = np.concatenate([y, [y_end[-1]]])
                
            return np.array(t), np.array(y)
            
        except Exception as e:
            print(f"混合求解器出错: {e}，使用备选方法")
            # 完全回退到备选求解器
            return self.fallback_solver.solve(t_span, y0, tol=tol, **kwargs)
    
    def step(self, t, y, h):
        """执行单步求解 - 动态选择最佳方法"""
        # 判断是否应该使用泰勒展开
        use_taylor = self._should_use_taylor_step(t, y, h)
        
        if use_taylor:
            try:
                return self.taylor_solver.taylor_step(t, y, h)
            except Exception as e:
                # 如果泰勒方法失败，回退到传统方法
                return self.fallback_solver.step(t, y, h)
        else:
            return self.fallback_solver.step(t, y, h)
    
    def _create_segments(self, t_span, max_length):
        """将时间区间分成多个段"""
        total_length = t_span[1] - t_span[0]
        num_segments = max(1, int(np.ceil(total_length / max_length)))
        
        segment_points = np.linspace(t_span[0], t_span[1], num_segments + 1)
        segments = [(segment_points[i], segment_points[i+1]) for i in range(num_segments)]
        
        return segments
    
    def _should_use_taylor(self, t, y, t_span, segment_idx, total_segments):
        """判断是否应该使用泰勒展开求解此段"""
        # 策略1: 在初始段更倾向于使用泰勒展开
        if segment_idx < total_segments * 0.3:
            return True
            
        # 策略2: 对于短区间，泰勒展开通常更高效
        if t_span[1] - t_span[0] < 0.1:
            return True
            
        # 策略3: 检查方程在此点的特性
        try:
            # 计算f在此点的值和一阶导数值
            f_val = self.f(t, y)
            
            # 估计雅可比矩阵的特征
            h = 1e-6
            if np.isscalar(y):
                df_val = (self.f(t, y + h) - f_val) / h
                
                # 对于变化较平滑的区域，泰勒展开表现更好
                if abs(df_val) < 10:
                    return True
            else:
                # 对于系统ODE，检查非刚性
                y_perturbed = y.copy()
                y_perturbed[0] += h
                df_val = (self.f(t, y_perturbed) - f_val) / h
                
                # 简单检查第一个分量的导数
                if abs(df_val[0]) < 10:
                    return True
        except:
            pass
            
        # 默认使用备选方法，泰勒方法仅在适合的情况下使用
        return False
    
    def _should_use_taylor_step(self, t, y, h):
        """判断单步计算时是否应该使用泰勒展开"""
        # 对于小步长，泰勒展开通常表现更好
        if h < 0.01:
            return True
        
        # 检查是否为系统ODE
        is_vector = hasattr(y, '__len__') and not isinstance(y, (str, bytes))
        
        # 检查此点的行为
        try:
            f_val = self.f(t, y)
            
            if is_vector:
                # 系统ODE - 使用范数评估
                return np.linalg.norm(f_val) < 100
            else:
                # 标量ODE
                return abs(f_val) < 100
        except:
            return False
        
        return False