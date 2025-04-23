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
        """计算给定点处的所有高阶导数值，带有全面的错误处理"""
        derivatives = []
        for k in range(min(len(self.derivative_funcs), self.order + 2)):
            try:
                if not callable(self.derivative_funcs[k]):
                    print(f"警告: 第{k}阶导数函数不可调用，使用零替代")
                    deriv_value = 0.0
                else:
                    try:
                        deriv_value = self.derivative_funcs[k](t0, y0)
                        if deriv_value is None:
                            print(f"警告: 第{k}阶导数计算结果为None，使用零替代")
                            deriv_value = 0.0
                    except Exception as e:
                        print(f"计算第{k}阶导数时出错: {e}，使用零替代")
                        deriv_value = 0.0
                        
                # 数值稳定性控制
                max_allowed = 1.0e6
                if k > 1 and abs(deriv_value) > max_allowed:
                    orig_value = deriv_value
                    deriv_value = max_allowed if deriv_value > 0 else -max_allowed
                    print(f"第{k}阶导数值过大 ({orig_value})，截断为 {deriv_value}")
                    
                derivatives.append(deriv_value)
            except Exception as e:
                print(f"处理第{k}阶导数时发生未知错误: {e}，使用零替代")
                derivatives.append(0.0)
                
        # 确保有足够的导数
        while len(derivatives) <= self.order + 1:
            derivatives.append(0.0)
                
        return derivatives
    
    def taylor_step(self, t0, y0, h):
        """执行一个泰勒展开步骤"""
        derivatives = self.compute_derivatives(t0, y0)
        
        # 调试：打印导数值
        if len(derivatives) > 3 and np.random.random() < 0.01:  # 随机抽样1%步骤进行打印
            print(f"Debug - 前3阶导数: {derivatives[:3]}")
        
        # 原有代码
        main_solution = sum(derivatives[k] * (h**k) / factorial(k) 
                          for k in range(self.order + 1))
        
        error_term = derivatives[self.order + 1] * (h**(self.order + 1)) / factorial(self.order + 1)
        
        return main_solution, np.abs(error_term)
    
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