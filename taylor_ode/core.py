import numpy as np
from sympy import symbols, diff, lambdify
import matplotlib.pyplot as plt
from numba import jit
from scipy.special import factorial  # 添加这行导入factorial函数

class TaylorODESolver:
    """基于泰勒展开的ODE求解器"""
    
    def __init__(self, f, order=5):
        """
        初始化泰勒展开求解器
        
        参数:
            f: 微分方程右端函数 f(t, y)
            order: 泰勒展开阶数
        """
        self.f = f
        self.order = order
        self._setup_symbolic_derivatives()
    
    def _setup_symbolic_derivatives(self):
        """使用符号计算设置高阶导数"""
        try:
            t, y = symbols('t y')
            
            # 存储符号表达式和函数
            self.symbolic_derivatives = [y]  # y^(0) = y
            self.derivative_funcs = [lambda t, y: y]  # 始终确保第0阶导数函数可用
            
            # 计算符号导数
            try:
                current_expr = self.f(t, y)
                self.symbolic_derivatives.append(current_expr)  # y^(1) = f(t, y)
                self.derivative_funcs.append(lambda t, y: self.f(t, y))  # 确保第1阶导数函数可用
                
                # 计算高阶导数
                for k in range(2, self.order + 2):
                    try:
                        # 对时间的全导数 dy^(k-1)/dt = ∂y^(k-1)/∂t + ∂y^(k-1)/∂y * f(t,y)
                        t_partial = diff(current_expr, t)
                        y_partial = diff(current_expr, y) * self.f(t, y)
                        current_expr = t_partial + y_partial
                        self.symbolic_derivatives.append(current_expr)
                        
                        # 立即转换为函数并验证
                        func = lambdify((t, y), current_expr, 'numpy')
                        if not callable(func):
                            raise ValueError(f"lambdify创建的第{k}阶导数函数不可调用")
                        
                        # 测试函数
                        try:
                            _ = func(0.0, 1.0)
                            self.derivative_funcs.append(func)
                        except Exception as e:
                            raise ValueError(f"第{k}阶导数函数测试失败: {e}")
                            
                    except Exception as e:
                        print(f"计算第{k}阶导数失败: {e}，使用数值近似")
                        # 退化到更简单的近似
                        self.derivative_funcs.append(self._make_fallback_deriv(k))
                        break
                        
            except Exception as e:
                print(f"符号微分过程中发生错误: {e}")
                # 确保所有导数函数被初始化
                while len(self.derivative_funcs) <= self.order + 1:
                    k = len(self.derivative_funcs)
                    self.derivative_funcs.append(self._make_fallback_deriv(k))
                    
        except Exception as e:
            print(f"符号微分完全失败: {e}")
            print("切换至纯数值差分模式...")
            
            # 重置导数函数列表
            self.derivative_funcs = []
            self.derivative_funcs.append(lambda t, y: y)  # 0阶导数
            self.derivative_funcs.append(lambda t, y: self.f(t, y))  # 1阶导数
            
            # 为高阶导数创建安全的近似函数
            for k in range(2, self.order + 2):
                self.derivative_funcs.append(self._make_fallback_deriv(k))

    def _make_fallback_deriv(self, order):
        """创建一个安全的后备导数函数"""
        if order == 0:
            return lambda t, y: y
        elif order == 1:
            return lambda t, y: self.f(t, y)
        elif order == 2:
            # 二阶导数特殊处理
            def second_deriv(t, y):
                try:
                    h = 1e-6
                    f1 = self.f(t, y)
                    y2 = y + h * f1
                    f2 = self.f(t + h, y2)
                    return (f2 - f1) / h
                except Exception:
                    # 使用更保守的估计
                    return 0.0
            return second_deriv
        else:
            # 高阶导数使用指数衰减的保守估计
            scale = 0.1**(order-2)
            return lambda t, y: scale * self.f(t, y) if callable(self.f) else 0.0

    def _setup_numeric_derivatives(self):
        """使用数值差分作为备选方案"""
        funcs = []
        
        # 0阶导数就是函数值本身
        funcs.append(lambda t, y: y)
        
        # 1阶导数是微分方程右侧
        funcs.append(lambda t, y: self.f(t, y))
        
        # 对于高阶导数，使用有限差分近似
        for k in range(2, self.order + 2):
            # 创建一个闭包来捕获k值
            def make_func(order):
                def numeric_deriv(t, y):
                    if order == 2:
                        # 2阶导数特殊处理
                        h = 1e-6
                        f1 = self.f(t, y)
                        y2 = y + h * f1
                        f2 = self.f(t + h, y2)
                        return (f2 - f1) / h
                    else:
                        # 高阶导数，使用中心差分
                        h = 1e-6
                        # 应用正则化避免数值爆炸
                        scale = 0.1**(order-2)  # 高阶导数衰减因子
                        return scale * self.f(t, y)  # 简化高阶导数估计
                return numeric_deriv
            
            funcs.append(make_func(k))
        
        return funcs
    
    def compute_derivatives(self, t0, y0):
        """计算泰勒展开所需的导数，带有增强的数值稳定性控制"""
        derivatives = []
        
        # 检查并确保导数函数可用
        missing_funcs = False
        for k in range(self.order + 2):
            if k >= len(self.derivative_funcs) or self.derivative_funcs[k] is None or not callable(self.derivative_funcs[k]):
                missing_funcs = True
                break
        
        # 如有必要，重新初始化导数函数
        if missing_funcs:
            print("警告: 导数函数有缺失或不可调用，重新初始化...")
            # 保存原始f函数
            original_f = self.f
            # 重置导数函数
            self.derivative_funcs = []
            self.derivative_funcs.append(lambda t, y: y)  # 0阶导数
            self.derivative_funcs.append(lambda t, y: original_f(t, y))  # 1阶导数
            
            # 为高阶导数创建更稳定的函数
            for k in range(2, self.order + 2):
                order = k
                # 随阶数增加减小数值，避免导数爆炸
                scale_factor = min(1.0, 1.0/(order*order)) if order > 3 else 1.0
                
                # 创建适当的导数函数
                if order == 2:
                    # 二阶导数使用数值差分
                    def deriv_2(t, y, f=original_f):
                        try:
                            h = min(1e-6, 0.01 * abs(y)) if y != 0 else 1e-6
                            f1 = f(t, y)
                            y2 = y + h * f1
                            f2 = f(t + h, y2)
                            return (f2 - f1) / h
                        except:
                            return scale_factor * f(t, y)
                    self.derivative_funcs.append(deriv_2)
                else:
                    # 高阶导数使用适应性衰减
                    def make_high_order(k, decay):
                        def high_order_deriv(t, y, f=original_f):
                            # 限制y的范围，避免数值极值
                            y_safe = max(min(y, 1e6), -1e6)
                            base_val = f(t, y_safe)
                            # 应用基于k的衰减
                            return base_val * decay * (0.7 ** (k-2))
                        return high_order_deriv
                    
                    self.derivative_funcs.append(make_high_order(order, scale_factor))
        
        # 计算所有导数
        for k in range(self.order + 2):
            try:
                if k >= len(self.derivative_funcs) or self.derivative_funcs[k] is None:
                    # 导数函数缺失
                    if k == 0:
                        deriv_value = y0
                    elif k == 1:
                        deriv_value = self.f(t0, y0)
                    else:
                        # 高阶导数使用适应性衰减
                        deriv_value = self.f(t0, y0) * 0.1 * (0.5 ** (k-2))
                elif not callable(self.derivative_funcs[k]):
                    # 导数函数不可调用
                    if k == 0:
                        deriv_value = y0
                    elif k == 1:
                        deriv_value = self.f(t0, y0)
                    else:
                        deriv_value = 0.0
                else:
                    # 计算导数值
                    try:
                        deriv_value = self.derivative_funcs[k](t0, y0)
                        if deriv_value is None:
                            # 结果为None
                            if k == 0:
                                deriv_value = y0
                            elif k == 1:
                                deriv_value = self.f(t0, y0)
                            else:
                                deriv_value = 0.0
                    except Exception as e:
                        # 计算出错
                        if k == 0:
                            deriv_value = y0
                        elif k == 1:
                            deriv_value = self.f(t0, y0)
                        else:
                            deriv_value = 0.0
                
                # 应用数值稳定性控制 - 使用更智能的阶数感知限制
                if k > 1:
                    # 基于阶数的最大允许值，高阶导数限制更严格
                    max_allowed = 1e6 / (10 ** (k-2)) if k > 2 else 1e6
                    if abs(deriv_value) > max_allowed:
                        # 截断但保持符号
                        deriv_value = max_allowed if deriv_value > 0 else -max_allowed
                
                derivatives.append(deriv_value)
            except Exception as e:
                # 完全失败的情况
                if k == 0:
                    derivatives.append(y0)
                elif k == 1:
                    derivatives.append(self.f(t0, y0))
                else:
                    derivatives.append(0.0)
        
        return derivatives
    
    def taylor_step(self, t0, y0, h):
        """执行一个泰勒展开步骤，带完整错误处理"""
        try:
            derivatives = self.compute_derivatives(t0, y0)
            
            # 随机调试输出
            if np.random.random() < 0.001:  # 随机抽样调试输出
                print(f"Debug - 前3阶导数: {derivatives[:3]}")
            
            # 计算泰勒展开
            main_solution = derivatives[0]  # 从0阶开始
            for k in range(1, self.order + 1):
                try:
                    # 使用scipy.special.factorial而不是np.factorial
                    term = derivatives[k] * (h**k) / factorial(k)
                    # 检查数值是否有效
                    if np.isfinite(term):
                        main_solution += term
                    else:
                        print(f"警告: 第{k}阶泰勒项非有限值，已忽略")
                except Exception as e:
                    print(f"计算第{k}阶项时出错: {e}")
            
            # 计算误差估计
            try:
                error_term = derivatives[self.order + 1] * (h**(self.order + 1)) / factorial(self.order + 1)
                error = np.abs(error_term)
                # 确保误差估计合理
                if not np.isfinite(error) or error > 1.0:
                    print(f"警告: 误差估计 {error} 不合理，使用保守值")
                    error = 1e-3 * abs(main_solution)  # 保守估计
            except Exception as e:
                print(f"计算误差估计时出错: {e}")
                error = 1e-3 * abs(main_solution)  # 保守估计
            
            return main_solution, error
        
        except Exception as e:
            print(f"泰勒步骤计算完全失败: {e}")
            # 退化为简单的欧拉步骤
            try:
                step = h * self.f(t0, y0)
                return y0 + step, abs(step) * 0.1
            except:
                print("欧拉步骤也失败，返回原始值")
                return y0, abs(y0) * 0.01
    
    def solve(self, t_span, y0, tol=1e-6, max_step=0.1, min_step=1e-8, fixed_points=None):
        """
        求解ODE，返回整个解曲线
        
        参数:
            t_span: 时间区间 [t0, t_end]
            y0: 初始状态
            tol: 容许误差
            max_step: 最大允许步长
            min_step: 最小允许步长
            fixed_points: 可选的固定时间点序列，如果提供则返回这些点上的解
            
        返回:
            (时间点数组, 解数组)
        """
        if fixed_points is not None:
            # 使用固定时间点计算解
            t_points = fixed_points
            y_points = []
            
            # 使用自适应求解获取高精度解
            t_adaptive, y_adaptive = self._solve_adaptive(t_span, y0, tol, max_step, min_step)
            
            # 对求解结果插值到指定时间点
            from scipy.interpolate import interp1d
            interp_func = interp1d(t_adaptive, y_adaptive, kind='cubic', bounds_error=False, fill_value="extrapolate")
            y_points = interp_func(t_points)
            
            return t_points, y_points
        else:
            # 原始自适应求解
            return self._solve_adaptive(t_span, y0, tol, max_step, min_step)
    
    def _solve_adaptive(self, t_span, y0, tol=1e-6, max_step=0.1, min_step=1e-8):
        """原始自适应步长求解算法的实现"""
        t_current = t_span[0]
        y_current = y0
        
        t_points = [t_current]
        y_points = [y_current]
        
        while t_current < t_span[1]:
            # 计算到终点的距离
            remaining = t_span[1] - t_current
            
            # 初始步长选择
            h = min(max_step, remaining)
            
            while True:
                # 尝试当前步长
                y_next, error = self.taylor_step(t_current, y_current, h)
                
                # 步长调整策略
                if error < tol:
                    # 接受当前步
                    y_current = y_next
                    t_current += h
                    
                    t_points.append(t_current)
                    y_points.append(y_current)
                    
                    # 增大下一步的步长 (但不超过max_step)
                    h = min(1.2 * h, max_step)
                    break
                else:
                    # 缩小步长重试
                    h_new = 0.9 * h * (tol / error)**(1.0 / (self.order + 1))
                    h = max(h_new, min_step)
                    
                    if h <= min_step:
                        raise RuntimeError(f"步长 {h} 已小于最小允许步长 {min_step}，无法满足精度要求")
        
        return np.array(t_points), np.array(y_points)
    
    def plot_solution(self, t, y, title="ODE Solution"):
        """绘制解曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(t, y, 'b-')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('y(t)')
        plt.grid(True)
        plt.show()


# 用于系统ODE的扩展
class TaylorSystemSolver:
    """用于ODE系统的泰勒展开求解器"""
    
    def __init__(self, f, order=5):
        """
        初始化系统求解器
        
        参数:
            f: 微分方程组右端函数 f(t, y) 其中y是向量
            order: 泰勒展开阶数
        """
        self.f = f
        self.order = order
        # 系统ODE通常使用自动微分而非符号计算
    
    def _finite_diff_jacobian(self, f, t, y, eps=1e-8):
        """使用有限差分计算雅可比矩阵"""
        n = len(y)
        jac = np.zeros((n, n))
        f0 = f(t, y)
        
        for i in range(n):
            y_perturbed = y.copy()
            y_perturbed[i] += eps
            f1 = f(t, y_perturbed)
            jac[:, i] = (f1 - f0) / eps
            
        return jac
    
    def compute_derivatives(self, t0, y0, max_order):
        """
        使用递推计算高阶导数
        
        递推公式:
        y^(n+1) = df^(n)/dt + df^(n)/dy * f(t,y)
        """
        n = len(y0)
        derivatives = [np.array(y0)]  # 0阶导数
        
        # 计算1阶导数
        f0 = self.f(t0, y0)
        derivatives.append(f0)
        
        # 递推计算高阶导数
        for k in range(2, max_order + 1):
            # 计算雅可比矩阵
            jac = self._finite_diff_jacobian(self.f, t0, y0)
            
            # 这里简化处理，假设f不显式依赖于t
            # 如果f显式依赖于t，需要额外计算∂f/∂t
            prev_deriv = derivatives[k-1]
            next_deriv = np.matmul(jac, prev_deriv)
            
            derivatives.append(next_deriv)
            
        return derivatives
    
    def taylor_step(self, t0, y0, h):
        """
        执行一个泰勒展开步骤
        
        参数:
            t0: 当前时间
            y0: 当前状态向量
            h: 步长
            
        返回:
            (下一状态, 误差估计)
        """
        derivatives = self.compute_derivatives(t0, y0, self.order + 1)
        
        # 计算主解 (order阶)
        main_solution = np.zeros_like(y0)
        for k in range(self.order + 1):
            # 将np.factorial替换为factorial
            main_solution += derivatives[k] * (h**k) / factorial(k)
        
        # 误差估计 (使用order+1阶项)
        # 将np.factorial替换为factorial
        error_term = derivatives[self.order + 1] * (h**(self.order + 1)) / factorial(self.order + 1)
        
        return main_solution, np.linalg.norm(error_term)
    
    def solve(self, t_span, y0, tol=1e-6, max_step=0.1, min_step=1e-8):
        """
        求解ODE系统，返回整个解曲线
        
        参数:
            t_span: 时间区间 [t0, t_end]
            y0: 初始状态向量
            tol: 容许误差
            max_step: 最大允许步长
            min_step: 最小允许步长
            
        返回:
            (时间点数组, 解数组)
        """
        t_current = t_span[0]
        y_current = np.array(y0)
        
        t_points = [t_current]
        y_points = [y_current]
        
        while t_current < t_span[1]:
            # 计算到终点的距离
            remaining = t_span[1] - t_current
            
            # 初始步长选择
            h = min(max_step, remaining)
            
            while True:
                # 尝试当前步长
                y_next, error = self.taylor_step(t_current, y_current, h)
                
                # 步长调整策略
                if error < tol:
                    # 接受当前步
                    y_current = y_next
                    t_current += h
                    
                    t_points.append(t_current)
                    y_points.append(y_current)
                    
                    # 增大下一步的步长 (但不超过max_step)
                    h = min(1.2 * h, max_step)
                    break
                else:
                    # 缩小步长重试
                    h_new = 0.9 * h * (tol / error)**(1.0 / (self.order + 1))
                    h = max(h_new, min_step)
                    
                    if h <= min_step:
                        raise RuntimeError(f"步长 {h} 已小于最小允许步长 {min_step}，无法满足精度要求")
        
        return np.array(t_points), np.array(y_points)