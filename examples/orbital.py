import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import datetime
from io import StringIO
from scipy.special import factorial  # 如果需要使用factorial，添加这行

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加父目录到搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from taylor_ode.core import TaylorSystemSolver

# 创建results主目录及orbital子目录
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
# 为该示例创建专用子目录
orbital_dir = os.path.join(results_dir, "orbital")
if not os.path.exists(orbital_dir):
    os.makedirs(orbital_dir)

# 引力常数 * 地球质量 (m^3/s^2)
GM = 3.986004418e14

def orbital_dynamics(t, state):
    """
    轨道动力学方程
    
    参数:
        t: 时间
        state: 状态向量 [x, y, z, vx, vy, vz]
        
    返回:
        导数 [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # 计算重力加速度
    ax = -GM * x / r**3
    ay = -GM * y / r**3
    az = -GM * z / r**3
    
    return np.array([vx, vy, vz, ax, ay, az])

def run_orbital_simulation():
    """执行卫星轨道预测"""
    # 创建输出缓冲区和结果文件
    output_buffer = StringIO()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(orbital_dir, f"orbital_results_{timestamp}.txt")
    data_file = os.path.join(orbital_dir, f"orbital_data_{timestamp}.csv")
    
    # 重定向标准输出到缓冲区
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    # 初始状态 (国际空间站轨道参数)
    # 位置单位: 米, 速度单位: 米/秒
    initial_state = np.array([6.78e6, 0, 0, 0, 7.68e3, 0])  # x,y,z,vx,vy,vz
    
    # 创建高阶泰勒求解器
    solver = TaylorSystemSolver(orbital_dynamics, order=8)
    
    # 求解5分钟内的轨道
    t_span = [0, 300]  # 秒
    tol = 1e-3  # 容许误差
    
    print("开始计算轨道...")
    t, states = solver.solve(t_span, initial_state, tol=tol, max_step=10.0)
    print(f"计算完成。使用了 {len(t)} 个时间步骤")
    
    # 提取位置数据
    positions = states[:, :3]
    
    # 保存位置数据到CSV文件
    with open(data_file, 'w') as f:
        f.write("time,x,y,z,vx,vy,vz\n")
        for i in range(len(t)):
            f.write(f"{t[i]},{states[i,0]},{states[i,1]},{states[i,2]},{states[i,3]},{states[i,4]},{states[i,5]}\n")
    
    print(f"轨道数据已保存到 {data_file}")
    
    # 计算起点和终点的距离
    distance = np.linalg.norm(positions[-1] - positions[0])
    print(f"5分钟后飞行距离: {distance/1000:.2f} 公里")
    
    # 估计位置误差 - 在实际应用中需要更精确的误差估计方法
    position_error = tol * np.linalg.norm(positions[-1])
    print(f"估计位置误差: {position_error:.2f} 米")
    
    # 步长统计
    step_sizes = np.diff(t)
    print(f"平均步长: {np.mean(step_sizes):.2f} 秒")
    print(f"最小步长: {np.min(step_sizes):.2f} 秒")
    print(f"最大步长: {np.max(step_sizes):.2f} 秒")
    
    # 可视化轨道
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制地球
    r_earth = 6.371e6  # 地球半径 (米)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_earth = r_earth * np.cos(u) * np.sin(v)
    y_earth = r_earth * np.sin(u) * np.sin(v)
    z_earth = r_earth * np.cos(v)
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.3)
    
    # 绘制轨道
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=2)
    
    # 标记起点和终点
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='起点')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='终点')
    
    # 设置图形属性
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('卫星轨道预测 (5分钟)')
    ax.legend()
    
    # 设置相等的坐标轴比例
    max_range = np.max([
        np.max(positions[:, 0]) - np.min(positions[:, 0]),
        np.max(positions[:, 1]) - np.min(positions[:, 1]),
        np.max(positions[:, 2]) - np.min(positions[:, 2])
    ])
    mid_x = (np.max(positions[:, 0]) + np.min(positions[:, 0])) / 2
    mid_y = (np.max(positions[:, 1]) + np.min(positions[:, 1])) / 2
    mid_z = (np.max(positions[:, 2]) + np.min(positions[:, 2])) / 2
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(orbital_dir, f"orbital_trajectory_{timestamp}.png"), dpi=300)
    
    # 保存不同视角的图像
    for elev in [20, 45, 90]:
        for azim in [0, 45, 90, 135]:
            ax.view_init(elev=elev, azim=azim)
            plt.savefig(os.path.join(orbital_dir, f"orbital_view_elev{elev}_azim{azim}_{timestamp}.png"), dpi=300)
    
    # 绘制步长自适应
    plt.figure(figsize=(10, 6))
    plt.plot(t[:-1], step_sizes, 'b-')
    plt.xlabel('时间 (秒)')
    plt.ylabel('步长 (秒)')
    plt.title('自适应步长变化')
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(orbital_dir, f"orbital_step_sizes_{timestamp}.png"), dpi=300)
    
    # 恢复标准输出并写入文件
    sys.stdout = original_stdout
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(output_buffer.getvalue())
        
    print(f"结果已保存到 {result_file}")
    print(f"图像和数据已保存到 {orbital_dir} 目录")
    
    # 显示所有图形
    plt.show()

if __name__ == "__main__":
    run_orbital_simulation()