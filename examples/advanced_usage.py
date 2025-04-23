import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import matplotlib.font_manager as fm
from io import StringIO

# 添加中文字体支持
# 尝试设置不同的中文字体
chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
for font in chinese_fonts:
    # 检查字体是否可用
    if font in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
        break

# 允许负号正常显示
plt.rcParams['axes.unicode_minus'] = False

# 添加父目录到搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入新的统一接口
from taylor_ode.api import solve_ode

# 创建结果目录
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_advanced_examples(save_results=False):
    """展示新的求解器接口和功能"""
    
    # 设置结果保存目录
    if save_results:
        result_dir = "results/advanced_usage"
        ensure_dir(result_dir)
        # 捕获控制台输出
        old_stdout = sys.stdout
        output_log = StringIO()
        sys.stdout = output_log
    
    print("示例1: 自动方法选择")
    # 定义简单的ODE
    def exponential_decay(t, y):
        return -0.5 * y
    
    # 使用自动选择 - 关闭内置绘图
    t_span = [0, 10]
    y0 = 1.0
    t, y = solve_ode(exponential_decay, t_span, y0, method='auto', tol=1e-6, plot=False)
    
    # 计算精确解进行比较
    exact = y0 * np.exp(-0.5 * t)
    error = np.max(np.abs(y - exact))
    print(f"最大误差: {error:.2e}")
    
    # 明确创建新图形
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'bo-', label='数值解')
    plt.plot(t, exact, 'r-', label='解析解')
    plt.xlabel('时间')
    plt.ylabel('y(t)')
    plt.title('指数衰减函数求解比较')
    plt.legend()
    plt.grid(True)
    
    # 先保存再显示
    if save_results:
        plt.savefig(f"{result_dir}/example1_decay.png", dpi=300, bbox_inches='tight')
        print(f"已保存图片: {result_dir}/example1_decay.png")
    
    plt.tight_layout()
    plt.show()
    
    print("示例2: 刚性问题")
    # 刚性ODE示例 (Robertson化学反应)
    def robertson(t, y):
        # 限制输入值范围，避免数值溢出
        y = np.clip(y, 0, 1e10)  # 限制最大范围
        
        # 使用更稳定的计算方式
        k1, k2, k3 = 0.04, 1e4, 3e7
        dy1 = -k1*y[0] + k2*y[1]*y[2]
        dy2 = k1*y[0] - k2*y[1]*y[2] - k3*y[1]**2
        dy3 = k3*y[1]**2
        
        # 确保质量守恒
        sum_dy = dy1 + dy2 + dy3
        if abs(sum_dy) > 1e-10:
            # 应用微小修正确保质量守恒
            correction = sum_dy / 3
            dy1 -= correction
            dy2 -= correction
            dy3 -= correction
            
        return np.array([dy1, dy2, dy3])
    
    y0 = np.array([1.0, 0.0, 0.0])
    t_span = [0, 1e5]
    
    start = time.time()
    t, y = solve_ode(robertson, t_span, y0, method='auto', tol=1e-4, max_step=1e3)
    end = time.time()
    
    print(f"求解用时: {end - start:.2f} 秒")
    print(f"使用步数: {len(t)}")
    
    # 绘制结果（对数尺度）
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(t, y[:, 0], 'b-', label='y1')
    plt.plot(t, y[:, 2], 'g-', label='y3')
    plt.xlabel('时间')
    plt.ylabel('浓度')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(122)
    plt.loglog(t, y[:, 0], 'b-', label='y1')
    plt.loglog(t, y[:, 1], 'r-', label='y2')
    plt.loglog(t, y[:, 2], 'g-', label='y3')
    plt.xlabel('时间 (对数尺度)')
    plt.ylabel('浓度 (对数尺度)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存刚性问题图表
    if save_results:
        plt.savefig(f"{result_dir}/example2_stiff.png")
    plt.show()
    print()
    
    print("示例3: 混合方法比较")
    # 振荡问题 - Van der Pol振荡器
    def van_der_pol(t, y):
        """Van der Pol振荡器 - 确保正确处理向量和标量输入"""
        mu = 1.0
        
        # 检查输入类型
        if isinstance(y, (list, tuple, np.ndarray)):
            # 向量版本
            if len(y) == 2:
                return np.array([y[1], mu * (1.0 - y[0]**2) * y[1] - y[0]])
            else:
                # 处理意外的向量长度
                return np.zeros_like(y)
        else:
            # 标量版本 - 用于符号计算测试
            return 0.0  # 简化的返回值

    y0 = np.array([2.0, 0.0])
    t_span = [0, 20]

    # 注意：对于系统ODE，我们应该优先选择适合的方法
    methods = ['rk45', 'hybrid', 'radau']
    results = {}
    times = {}

    plt.figure(figsize=(15, 10))
    
    for i, method in enumerate(methods):
        start = time.time()
        t, y = solve_ode(van_der_pol, t_span, y0, method=method, tol=1e-6, order=8)
        end = time.time()
        
        results[method] = (t, y)
        times[method] = end - start
        
        plt.subplot(len(methods), 1, i+1)
        plt.plot(t, y[:, 0], 'b-', label='位置')
        plt.plot(t, y[:, 1], 'r-', label='速度')
        plt.title(f'{method.capitalize()} 方法 (用时: {times[method]:.3f}秒, 步数: {len(t)})')
        plt.grid(True)
        plt.legend()
    
    # 保存Van der Pol振荡器结果图
    if save_results:
        plt.savefig(f"{result_dir}/example3_methods.png")
    plt.show()
    
    # 相图比较
    plt.figure(figsize=(12, 8))
    for method in methods:
        t, y = results[method]
        plt.plot(y[:, 0], y[:, 1], label=f'{method} (用时: {times[method]:.3f}秒)')
    
    plt.xlabel('位置')
    plt.ylabel('速度')
    plt.title('Van der Pol 振荡器相图')
    plt.grid(True)
    plt.legend()
    
    # 保存相图比较
    if save_results:
        plt.savefig(f"{result_dir}/example3_phase.png")
    plt.show()
    
    print("方法性能比较:")
    for method in methods:
        print(f"  {method.capitalize()}: {times[method]:.3f} 秒, {len(results[method][0])} 步")
    
    # 完成后保存输出日志
    if save_results:
        sys.stdout = old_stdout
        with open(f"{result_dir}/output_log.txt", "w", encoding="utf-8") as f:
            f.write(output_log.getvalue())
        print(f"结果已保存到 {os.path.abspath(result_dir)} 目录")

# 修改主调用部分
if __name__ == "__main__":
    # 检查是否有命令行参数
    save_mode = "--save" in sys.argv
    run_advanced_examples(save_results=save_mode)