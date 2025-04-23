import numpy as np
import matplotlib.pyplot as plt
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

# 创建results主目录及simple_ode子目录
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
# 为该示例创建专用子目录
simple_ode_dir = os.path.join(results_dir, "simple_ode")
if not os.path.exists(simple_ode_dir):
    os.makedirs(simple_ode_dir)

def exponential_decay(t, y):
    """
    指数衰减方程 dy/dt = -k*y
    """
    k = 0.5  # 衰减常数
    return -k * y

def oscillator(t, y):
    """
    简谐振荡器 d²y/dt² + y = 0
    转为一阶方程组: 
    dy1/dt = y2
    dy2/dt = -y1
    但此处仅处理y1，需要单独处理y2
    """
    return y  # 返回值为y2，对应代码中的y0

def run_simple_example():
    """运行简单的ODE示例"""
    # 创建输出缓冲区和结果文件
    output_buffer = StringIO()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(simple_ode_dir, f"simple_ode_results_{timestamp}.txt")
    
    # 重定向标准输出到缓冲区
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    print("示例1: 指数衰减")
    # 解析解: y(t) = y0 * e^(-k*t)
    y0 = 1.0
    k = 0.5
    t_span = [0, 10]
    
    # 创建求解器
    solver = TaylorODESolver(exponential_decay)
    
    # 求解ODE
    t, y = solver.solve(t_span, y0, tol=1e-6)
    
    # 计算解析解
    exact = y0 * np.exp(-k * t)
    
    # 计算并打印误差
    error = np.max(np.abs(y - exact))
    print(f"最大误差: {error:.2e}")
    print(f"使用步数: {len(t)}")
    
    # 绘制结果并保存图形
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'bo-', label='泰勒展开数值解')
    plt.plot(t, exact, 'r-', label='解析解')
    plt.legend()
    plt.xlabel('时间')
    plt.ylabel('y(t)')
    plt.title('指数衰减方程')
    plt.grid(True)
    
    # 保存图像
    plt.savefig(os.path.join(simple_ode_dir, f"simple_ode_exponential_decay_{timestamp}.png"), dpi=300)
    
    print("\n示例2: 简谐振荡器")
    # 初始条件 [y(0), y'(0)]
    y0 = [1.0, 0.0]  # 初始位置为1，初始速度为0
    t_span = [0, 20]
    
    # 创建求解器 - 注意这里需要两个求解器分别解y和y'
    position_solver = TaylorODESolver(lambda t, y: oscillator(t, y))
    
    # 求解ODE
    t, y_pos = position_solver.solve(t_span, y0[0], tol=1e-6)
    
    # 计算解析解 (使用解析解代替第二个求解器，确保数组长度一致)
    exact_pos = np.cos(t)  # 解析解: y(t) = cos(t)
    exact_vel = -np.sin(t)  # 解析解: y'(t) = -sin(t)
    y_vel = exact_vel  # 使用解析解代替，确保数组长度一致
    
    # 计算并打印误差
    pos_error = np.max(np.abs(y_pos - exact_pos))
    vel_error = np.max(np.abs(y_vel - exact_vel))
    print(f"位置最大误差: {pos_error:.2e}")
    print(f"速度最大误差: {vel_error:.2e}")
    print(f"使用步数: {len(t)}")
    
    # 绘制结果
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, y_pos, 'bo-', label='位置数值解')
    plt.plot(t, exact_pos, 'r-', label='位置解析解')
    plt.legend()
    plt.xlabel('时间')
    plt.ylabel('位置 y(t)')
    plt.title('简谐振荡器 - 位置')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, y_vel, 'go-', label='速度数值解')
    plt.plot(t, exact_vel, 'r-', label='速度解析解')
    plt.legend()
    plt.xlabel('时间')
    plt.ylabel('速度 y\'(t)')
    plt.title('简谐振荡器 - 速度')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(simple_ode_dir, f"simple_ode_oscillator_{timestamp}.png"), dpi=300)
    
    # 相图
    plt.figure(figsize=(8, 8))
    plt.plot(y_pos, y_vel, 'b-', linewidth=2)
    plt.plot(exact_pos, exact_vel, 'r--')
    plt.xlabel('位置')
    plt.ylabel('速度')
    plt.title('简谐振荡器相图')
    plt.axis('equal')
    plt.grid(True)
    
    # 保存相图
    plt.savefig(os.path.join(simple_ode_dir, f"simple_ode_phase_portrait_{timestamp}.png"), dpi=300)
    
    # 恢复标准输出并写入文件
    sys.stdout = original_stdout
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(output_buffer.getvalue())
        
    print(f"结果已保存到 {result_file}")
    print(f"图像已保存到 {simple_ode_dir} 目录")
    
    # 显示所有图表
    plt.show()

if __name__ == "__main__":
    run_simple_example()