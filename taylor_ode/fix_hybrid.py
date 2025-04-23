"""修复HybridTaylorSolver类中的方法调用问题"""
import numpy as np
from taylor_ode.solvers import HybridTaylorSolver

# 保存原始方法
_original_solve = HybridTaylorSolver.solve

# 修复后的solve方法
def patched_solve(self, t_span, y0, tol=1e-6, **kwargs):
    """修复后的solve方法，正确处理单步计算"""
    # 检查是否是系统ODE
    is_vector = hasattr(y0, '__len__') and len(y0) > 1 and not np.isscalar(y0[0])
    
    # 对于系统ODE，优先使用fallback_solver
    if is_vector:
        print(f"系统ODE检测：优先使用{self.fallback_method}求解")
        try:
            return self.fallback_solver.solve(t_span, y0, tol=tol, **kwargs)
        except Exception as e:
            print(f"备选求解器失败: {e}")
            from scipy.integrate import solve_ivp
            sol = solve_ivp(self.f, t_span, y0, method='RK45', rtol=tol, atol=tol)
            return sol.t, sol.y.T
    
    # 修复单步计算过程
    try:
        # 初始化计算参数
        t_current = t_span[0]
        y_current = y0
        t_points = [t_current]
        y_points = [y_current]
        
        max_step = kwargs.get('max_step', 0.1)
        min_step = kwargs.get('min_step', 1e-10)
        
        while t_current < t_span[1]:
            # 计算剩余距离
            distance_to_end = t_span[1] - t_current
            # 调整步长不超过终点
            h = min(max_step, distance_to_end)
            
            # 这里是关键修复：适应taylor_step返回两个或三个值的情况
            if self._should_use_taylor_step(t_current, y_current, h):
                try:
                    # 尝试获取taylor_step的结果，并正确处理不同数量的返回值
                    result = self.taylor_solver.taylor_step(t_current, y_current, h)
                    
                    # 根据返回值数量不同处理
                    if isinstance(result, tuple):
                        if len(result) == 3:
                            y_next, error, h_used = result
                        elif len(result) == 2:
                            y_next, error = result
                            h_used = h  # 如果没有返回h_used，就使用原始步长
                        else:
                            raise ValueError(f"Unexpected return value length: {len(result)}")
                    else:
                        # 如果不是元组，假设只返回了y_next
                        y_next = result
                        error = 0.0
                        h_used = h
                    
                    t_current += h_used
                    y_current = y_next
                except Exception as e:
                    print(f"泰勒步长计算出错: {e}，切换到备选方法")
                    # 使用fallback求解剩余部分
                    remaining_span = [t_current, t_span[1]]
                    t_rest, y_rest = self.fallback_solver.solve(remaining_span, y_current, tol=tol)
                    
                    # 合并结果
                    t_points = np.append(t_points, t_rest[1:])
                    y_points = np.append(y_points, y_rest[1:])
                    return t_points, y_points
            else:
                # 使用备选求解器计算剩余部分
                remaining_span = [t_current, t_span[1]]
                t_rest, y_rest = self.fallback_solver.solve(remaining_span, y_current, tol=tol)
                
                # 合并结果
                t_points = np.append(t_points, t_rest[1:])
                y_points = np.append(y_points, y_rest[1:])
                return t_points, y_points
            
            # 记录当前点
            t_points.append(t_current)
            y_points.append(y_current)
            
            # 检查是否达到或接近终点
            if abs(t_current - t_span[1]) < min_step:
                break
        
        return np.array(t_points), np.array(y_points)
        
    except Exception as e:
        print(f"混合求解器出错: {e}，使用备选方法")
        # 完全回退到备选求解器
        return self.fallback_solver.solve(t_span, y0, tol=tol, **kwargs)

# 应用补丁
HybridTaylorSolver.solve = patched_solve

# 修复_should_use_taylor_step方法确保其存在
if not hasattr(HybridTaylorSolver, '_should_use_taylor_step'):
    def _should_use_taylor_step(self, t, y, h):
        """判断是否应该使用泰勒步长"""
        # 简单策略：步长小时使用泰勒展开
        if h < 0.05:
            return True
            
        # 对于正常区间，使用泰勒的概率为70%
        return np.random.random() < 0.7
        
    HybridTaylorSolver._should_use_taylor_step = _should_use_taylor_step

print("已应用增强版HybridTaylorSolver修复")