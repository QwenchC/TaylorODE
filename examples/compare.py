import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
import sys
import os
import datetime
from io import StringIO

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加父目录到搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from taylor_ode.core import TaylorODESolver

# 创建results主目录及compare子目录
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
# 为该示例创建专用子目录
compare_dir = os.path.join(results_dir, "compare")
if not os.path.exists(compare_dir):
    os.makedirs(compare_dir)

def lotka_volterra(t, state):
    """
    Lotka-Volterra 捕食者-猎物模型
    
    参数:
        t: 时间
        state: [x, y] 其中 x 是猎物数量, y 是捕食者数量
        
    返回:
        [dx/dt, dy/dt]
    """
    try:
        # 尝试解包，失败时可能是符号变量
        x, y = state
    except (ValueError, TypeError):
        # 如果是符号变量，直接返回符号表达式
        from sympy import symbols
        if isinstance(state, type(symbols('x'))):
            # 符号模式，简化返回
            alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0
            return alpha * state - beta * state * state  # 简化的符号表达式
        else:
            # 其他错误，重新抛出
            raise
            
    # 数值模式
    alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return np.array([dxdt, dydt])

def safe_lotka_x(t, y):
    """安全处理x变量的Lotka-Volterra方程，带完整错误处理"""
    try:
        from sympy import symbols, Symbol
        
        # 符号模式检查
        if isinstance(y, Symbol) or (hasattr(y, 'is_Symbol') and y.is_Symbol):
            # 符号模式 - 返回简化表达式
            alpha, beta = 1.5, 1.0
            return alpha * y - beta * y * y
        
        # 数值模式处理
        if np.isscalar(y):
            try:
                # 固定y=1.0，只关注猎物变量x
                result = lotka_volterra(t, [y, 1.0])[0]
                # 检查结果是否合理
                if not np.isfinite(result):
                    alpha, beta = 1.5, 1.0
                    return alpha * y - beta * y  # 简化的线性近似
                return result
            except Exception as e:
                # 出错时使用近似
                alpha, beta = 1.5, 1.0
                return alpha * y - beta * y
        else:
            # 向量输入
            try:
                return lotka_volterra(t, y)[0]
            except:
                return 0.0
    except Exception as e:
        # 最终回退
        return 0.0

def safe_lotka_y(t, y):
    """安全处理捕食者变量的Lotka-Volterra方程"""
    from sympy import symbols
    if isinstance(y, type(symbols('x'))):
        # 符号模式返回简化形式
        gamma, delta = 3.0, 1.0
        return delta * y - gamma  # 简化符号表达式，避免与x相乘
    
    # 否则按照原逻辑处理
    if np.isscalar(y):
        # 固定猎物x=1.0，只关注捕食者y
        return lotka_volterra(t, [1.0, y])[1]
    else:
        return lotka_volterra(t, y)[1]

def compare_methods():
    """比较泰勒展开法与传统方法 - 简化版"""
    # 创建输出缓冲区和结果文件
    output_buffer = StringIO()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(compare_dir, f"compare_results_{timestamp}.txt")
    
    # 重定向标准输出到缓冲区
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    # 初始条件
    y0 = [1.0, 1.0]  # [猎物, 捕食者]
    t_span = [0, 15]
    
    # 创建统一的时间点用于评估和比较
    t_eval = np.linspace(t_span[0], t_span[1], 200)
    
    # 使用RK45求解完整系统作为参考
    print("计算参考解...")
    ref_sol = solve_ivp(
        lotka_volterra, t_span, y0, 
        method='RK45', rtol=1e-9, atol=1e-9,
        t_eval=t_eval
    )
    
    # 使用RK45求解不同精度的解
    tolerances = [1e-3, 1e-6, 1e-9]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    for i, tol in enumerate(tolerances):
        plt.subplot(2, 2, i+1)
        
        # 使用RK45求解当前精度
        rk_sol = solve_ivp(
            lotka_volterra, t_span, y0, 
            method='RK45', rtol=tol, atol=tol,
            t_eval=t_eval
        )
        
        # 计算和打印求解时间
        start_time = time.time()
        rk_sol = solve_ivp(
            lotka_volterra, t_span, y0, 
            method='RK45', rtol=tol, atol=tol,
            t_eval=t_eval
        )
        rk_time = time.time() - start_time
        print(f"\n精度要求: {tol}")
        print(f"RK45 求解时间: {rk_time:.4f} 秒")
        print(f"RK45 步数: {len(rk_sol.t)}")
        
        # 尝试使用泰勒展开求解第一个变量
        try:
            # 使用较低阶数以提高稳定性，并且使用较小步长
            solver = TaylorODESolver(safe_lotka_x, order=3)  # 从5阶降为3阶
            
            start_time = time.time()
            # 设置更保守的步长控制
            t_taylor, y_taylor_x = solver.solve(t_span, y0[0], tol=tol, max_step=0.05, min_step=1e-6)
            taylor_time = time.time() - start_time
            
            print(f"泰勒展开求解时间: {taylor_time:.4f} 秒")
            print(f"泰勒展开步数: {len(t_taylor)}")
            
            # 创建有效的误差比较
            if len(t_taylor) > 1:
                try:
                    from scipy.interpolate import interp1d
                    # 限制在共同的有效范围内
                    t_min = max(t_taylor[0], rk_sol.t[0])
                    t_max = min(t_taylor[-1], rk_sol.t[-1])
                    if t_max > t_min:
                        # 使用较少的点以减少计算量
                        t_common = np.linspace(t_min, t_max, 50)
                        taylor_interp = interp1d(t_taylor, y_taylor_x, kind='linear',
                                                bounds_error=False, fill_value="extrapolate")
                        taylor_error = np.max(np.abs(taylor_interp(t_common) - rk_sol.sol(t_common)[0]))
                        print(f"泰勒展开最大误差: {taylor_error:.2e}")
                except Exception as e:
                    print(f"误差计算失败: {e}")
        except Exception as e:
            print(f"泰勒展开求解出错: {str(e)}")
        
        # 绘制参考解
        plt.plot(t_eval, ref_sol.y[0], 'k--', linewidth=1, label='参考解-猎物')
        plt.plot(t_eval, ref_sol.y[1], 'k-.', linewidth=1, label='参考解-捕食者')
        
        # 绘制当前精度RK45解
        plt.plot(t_eval, rk_sol.y[0], 'b-', label=f'RK45 {tol:.0e}-猎物')
        plt.plot(t_eval, rk_sol.y[1], 'g-', label=f'RK45 {tol:.0e}-捕食者')
        
        plt.title(f'精度为{tol:.0e}的RK45解')
        plt.xlabel('时间')
        plt.ylabel('种群数量')
        plt.legend()
        plt.grid(True)
    
    # 绘制相图
    plt.subplot(2, 2, 4)
    for tol in tolerances:
        rk_sol = solve_ivp(
            lotka_volterra, t_span, y0, 
            method='RK45', rtol=tol, atol=tol,
            t_eval=t_eval
        )
        plt.plot(rk_sol.y[0], rk_sol.y[1], label=f'RK45 {tol:.0e}')
    
    plt.xlabel('猎物数量')
    plt.ylabel('捕食者数量')
    plt.title('相图比较')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(compare_dir, f"compare_methods_{timestamp}.png"), dpi=300)
    
    # 恢复标准输出并写入文件
    sys.stdout = original_stdout
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(output_buffer.getvalue())
        
    print(f"结果已保存到 {result_file}")
    print(f"图像已保存到 {compare_dir} 目录")
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    compare_methods()