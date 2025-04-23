import os
import sys

# 添加项目根目录到Python搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置matplotlib支持中文显示
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'Arial Unicode MS']  # 优先使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'

# 检查字体是否正确加载
print("当前系统可用字体:", [f for f in matplotlib.font_manager.findSystemFonts() if os.path.basename(f).startswith(('sim', 'Sim', 'micro', 'Micro'))])

# 导入其他模块
import time
import pandas as pd
from math import factorial
from scipy.integrate import solve_ivp

# 现在可以导入taylor_ode模块了
from taylor_ode.core import TaylorODESolver
from taylor_ode.solvers import TaylorSolver, RK45Solver, RadauSolver, HybridTaylorSolver
from taylor_ode.utils import analyze_ode

# 导入兼容函数
from taylor_ode.math_compat import compatible_sin, compatible_cos, compatible_exp

# 创建结果目录
import os
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir("results/taylor_evaluation")

def evaluate_taylor_order_effects():
    """评估不同阶数泰勒展开的效果"""
    print("评估1: 不同阶数泰勒展开的效果")
    
    # 简单ODE: 指数衰减
    def exp_decay(t, y):
        return -0.5 * y
    
    # 使用不同阶数求解
    t_span = [0, 10]
    y0 = 1.0
    orders = [2, 3, 5, 8, 10]
    results = {}
    
    # 计算精确解用于比较
    t_eval = np.linspace(t_span[0], t_span[1], 100)
    exact = y0 * np.exp(-0.5 * t_eval)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 求解并记录性能、误差
    performance_data = []
    for order in orders:
        print(f"  计算阶数 {order}...")
        solver = TaylorODESolver(exp_decay, order=order)
        
        # 计时求解
        start_time = time.time()
        t, y = solver.solve(t_span, y0, tol=1e-9, max_step=0.1)
        solve_time = time.time() - start_time
        
        # 计算误差 (需要插值到相同的时间点)
        from scipy.interpolate import interp1d
        interp_func = interp1d(t, y, kind='cubic', bounds_error=False, fill_value="extrapolate")
        y_interp = interp_func(t_eval)
        error = np.abs(y_interp - exact)
        max_error = np.max(error)
        avg_error = np.mean(error)
        
        # 保存结果
        results[order] = (t, y, solve_time, max_error)
        performance_data.append([order, solve_time, len(t), max_error, avg_error])
        
        # 绘制解
        axes[0].plot(t, y, '.-', label=f'泰勒{order}阶')
    
    # 绘制精确解
    axes[0].plot(t_eval, exact, 'k--', label='精确解')
    axes[0].set_xlabel('时间')
    axes[0].set_ylabel('y(t)')
    axes[0].set_title('不同阶数泰勒展开的解')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制误差比较
    for order in orders:
        t, y, _, _ = results[order]
        interp_func = interp1d(t, y, kind='cubic', bounds_error=False, fill_value="extrapolate")
        y_interp = interp_func(t_eval)
        error = np.abs(y_interp - exact)
        axes[1].semilogy(t_eval, error, label=f'泰勒{order}阶')
    
    axes[1].set_xlabel('时间')
    axes[1].set_ylabel('绝对误差 (对数)')
    axes[1].set_title('不同阶数泰勒展开的误差比较')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("results/taylor_evaluation/order_effects.png")
    plt.show()
    
    # 创建性能表格
    df = pd.DataFrame(performance_data, columns=['阶数', '求解时间(秒)', '步数', '最大误差', '平均误差'])
    print(df.to_string(index=False))
    
    # 保存结果
    with open("results/taylor_evaluation/order_performance.txt", "w", encoding='utf-8') as f:
        f.write(df.to_string(index=False))
    
    return df

def visualize_taylor_expansion_terms():
    """可视化泰勒展开中各阶项的贡献"""
    print("\n评估2: 泰勒展开各阶项的贡献")
    
    # 定义测试ODE
    def nonlinear_ode(t, y):
        """使用兼容函数的非线性振荡"""
        return compatible_sin(t) - 0.1 * y**2
    
    # 创建求解器
    solver = TaylorODESolver(nonlinear_ode, order=8)
    
    # 选择一个时间点和状态计算泰勒项
    t0 = 1.0
    y0 = 2.0
    h = 0.1  # 步长
    
    # 计算各阶导数
    derivatives = solver.compute_derivatives(t0, y0)
    
    # 计算各阶泰勒项
    taylor_terms = []
    cumulative_sum = y0
    
    for k in range(1, len(derivatives)):
        term = derivatives[k] * (h**k) / factorial(k)
        cumulative_sum += term
        taylor_terms.append((k, term, cumulative_sum))
    
    # 计算"真实"下一步值作为参考
    t_span = [t0, t0 + h]
    sol = solve_ivp(nonlinear_ode, t_span, [y0], method='RK45', rtol=1e-12, atol=1e-12)
    true_next = sol.y[0][-1]
    
    # 绘制各阶项的贡献
    plt.figure(figsize=(14, 8))
    
    # 上半图：各阶项贡献大小
    plt.subplot(2, 1, 1)
    ks, terms, _ = zip(*taylor_terms)
    plt.bar(ks, np.abs(terms), alpha=0.7)
    plt.yscale('log')
    plt.xlabel('泰勒展开阶数')
    plt.ylabel('项贡献大小 (对数)')
    plt.title(f'泰勒展开各阶项贡献 (t={t0}, y={y0}, h={h})')
    plt.grid(True)
    
    # 下半图：累积和与真实值比较
    plt.subplot(2, 1, 2)
    _, _, sums = zip(*taylor_terms)
    plt.plot(ks, sums, 'bo-', label='泰勒展开累积和')
    plt.axhline(y=true_next, color='r', linestyle='--', label=f'真实下一步值: {true_next:.8f}')
    plt.xlabel('累积到阶数')
    plt.ylabel('y 值')
    plt.title('泰勒展开累积和收敛性')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/taylor_evaluation/taylor_terms.png")
    plt.show()
    
    # 打印各阶贡献
    print("\n各阶泰勒项贡献:")
    print(f"初始值 y0 = {y0}")
    accumulated = y0
    for k, term, cum_sum in taylor_terms:
        print(f"阶数 {k}: 贡献 = {term:.8e}, 累积和 = {cum_sum:.8f}, 相对贡献 = {abs(term/y0)*100:.6f}%")
        accumulated += term
    
    print(f"\n真实下一步值: {true_next:.8f}")
    print(f"泰勒近似最终值: {accumulated:.8f}")
    print(f"近似误差: {abs(accumulated - true_next):.8e}")
    
    # 保存结果
    terms_data = [["初始值", "", y0, "100%"]]
    for k, term, cum_sum in taylor_terms:
        terms_data.append([k, term, cum_sum, f"{abs(term/y0)*100:.6f}%"])
    terms_data.append(["真实值", "", true_next, ""])
    terms_df = pd.DataFrame(terms_data, columns=['阶数', '贡献', '累积和', '相对贡献'])
    
    with open("results/taylor_evaluation/taylor_terms.txt", "w", encoding='utf-8') as f:
        f.write(terms_df.to_string(index=False))
    
    return terms_df

def compare_taylor_with_other_methods():
    """比较泰勒展开与其他方法在不同类型问题上的表现"""
    print("\n评估3: 泰勒展开与其他方法比较")
    
    # 定义几种不同类型的ODE
    problems = {
        "简单ODE": {
            "name": "指数衰减",
            "func": lambda t, y: -0.5 * y,
            "t_span": [0, 10],
            "y0": 1.0,
            "exact": lambda t: np.exp(-0.5 * t)
        },
        "非线性ODE": {
            "name": "非线性振荡",
            "func": lambda t, y: compatible_sin(t) - 0.1 * y**2,
            "t_span": [0, 10],
            "y0": 0.5,
            "exact": None  # 无解析解
        },
        "刚性ODE": {
            "name": "刚性微分方程",
            "func": lambda t, y: -15 * y + 8 * compatible_sin(t),
            "t_span": [0, 10],
            "y0": 0.0,
            "exact": lambda t: (8/17) * (np.sin(t) - (15/8) * np.cos(t) + (15/8) * np.exp(-15*t))
        },
        "系统ODE": {
            "name": "Van der Pol",
            "func": lambda t, y: np.array([y[1], (1 - y[0]**2) * y[1] - y[0]]),
            "t_span": [0, 10],
            "y0": np.array([2.0, 0.0]),
            "exact": None  # 无解析解
        }
    }
    
    # 定义要比较的方法
    methods = {
        "Taylor3": {"solver": TaylorSolver, "params": {"order": 3}},
        "Taylor5": {"solver": TaylorSolver, "params": {"order": 5}},
        "Taylor8": {"solver": TaylorSolver, "params": {"order": 8}},
        "RK45": {"solver": RK45Solver, "params": {}},
        "Radau": {"solver": RadauSolver, "params": {}},
        "Hybrid": {"solver": HybridTaylorSolver, "params": {"order": 5}}
    }
    
    # 误差容限
    tolerances = [1e-3, 1e-6, 1e-9]
    
    # 结果收集
    all_results = []
    
    # 对每种问题类型评估
    for prob_key, problem in problems.items():
        print(f"\n  评估问题: {problem['name']}")
        
        # 创建一个大图比较所有方法
        fig, axes = plt.subplots(len(tolerances), 2, figsize=(15, 5*len(tolerances)))
        
        for tol_idx, tol in enumerate(tolerances):
            print(f"    误差容限: {tol}")
            results_for_tol = {'问题': problem['name'], '容限': tol}
            
            for method_name, method_info in methods.items():
                # 对于系统ODE，跳过不适合的求解器
                if prob_key == "系统ODE" and method_name.startswith("Taylor") and not method_name.startswith("Taylor8"):
                    results_for_tol[f'{method_name}_时间'] = 'N/A'
                    results_for_tol[f'{method_name}_步数'] = 'N/A'
                    results_for_tol[f'{method_name}_误差'] = 'N/A'
                    continue
                
                # 创建求解器
                try:
                    solver = method_info["solver"](problem["func"], **method_info["params"])
                    
                    # 计时求解
                    start_time = time.time()
                    t, y = solver.solve(problem["t_span"], problem["y0"], tol=tol)
                    solve_time = time.time() - start_time
                    
                    # 记录结果
                    results_for_tol[f'{method_name}_时间'] = f"{solve_time:.4f}"
                    results_for_tol[f'{method_name}_步数'] = len(t)
                    
                    # 计算误差（如果有精确解）
                    if problem["exact"] is not None:
                        if not isinstance(problem["y0"], np.ndarray):
                            exact_y = problem["exact"](t)
                            error = np.max(np.abs(y - exact_y))
                            results_for_tol[f'{method_name}_误差'] = f"{error:.2e}"
                        else:
                            results_for_tol[f'{method_name}_误差'] = 'N/A'  # 系统ODE无法直接计算误差
                    else:
                        results_for_tol[f'{method_name}_误差'] = 'N/A'
                    
                    # 绘制解
                    if not isinstance(problem["y0"], np.ndarray):
                        axes[tol_idx, 0].plot(t, y, '-', label=method_name)
                    else:
                        # 系统ODE绘制第一个分量
                        axes[tol_idx, 0].plot(t, y[:, 0], '-', label=f"{method_name}")
                
                except Exception as e:
                    print(f"      {method_name} 失败: {e}")
                    results_for_tol[f'{method_name}_时间'] = 'Error'
                    results_for_tol[f'{method_name}_步数'] = 'Error'
                    results_for_tol[f'{method_name}_误差'] = 'Error'
            
            # 绘制精确解（如果有）
            if problem["exact"] is not None:
                t_exact = np.linspace(problem["t_span"][0], problem["t_span"][1], 1000)
                y_exact = problem["exact"](t_exact)
                axes[tol_idx, 0].plot(t_exact, y_exact, 'k--', label='精确解')
            
            # 图表标题和标签
            axes[tol_idx, 0].set_title(f'{problem["name"]} (tol={tol})')
            axes[tol_idx, 0].set_xlabel('时间')
            axes[tol_idx, 0].set_ylabel('y(t)')
            axes[tol_idx, 0].legend()
            axes[tol_idx, 0].grid(True)
            
            # 绘制性能对比
            method_names = [m for m in methods.keys() if f'{m}_时间' in results_for_tol and results_for_tol[f'{m}_时间'] != 'N/A' and results_for_tol[f'{m}_时间'] != 'Error']
            times = [float(results_for_tol[f'{m}_时间']) for m in method_names]
            steps = [results_for_tol[f'{m}_步数'] for m in method_names]
            
            if method_names:
                bars = axes[tol_idx, 1].bar(method_names, times)
                axes[tol_idx, 1].set_title(f'求解时间对比 (tol={tol})')
                axes[tol_idx, 1].set_ylabel('时间 (秒)')
                
                # 显示步数
                for i, v in enumerate(steps):
                    if v != 'N/A' and v != 'Error':
                        axes[tol_idx, 1].text(i, times[i] + 0.001, f"{v}步", 
                                             ha='center', va='bottom', rotation=0)
            
            all_results.append(results_for_tol)
        
        plt.tight_layout()
        plt.savefig(f"results/taylor_evaluation/{prob_key}_methods_comparison.png")
        plt.show()
    
    # 创建结果表格
    results_df = pd.DataFrame(all_results)
    print("\n方法比较结果:")
    print(results_df.to_string(index=False))
    
    # 保存结果
    with open("results/taylor_evaluation/methods_comparison.txt", "w", encoding='utf-8') as f:
        f.write(results_df.to_string(index=False))
    
    return results_df

def analyze_step_size_adaptation():
    """分析泰勒展开的步长自适应情况"""
    print("\n评估4: 步长自适应分析")
    
    # 定义一个有快速变化区域的ODE
    def variable_stiffness(t, y):
        return -5 * y + 10 * np.sin(t) - 50 * np.exp(-10 * (t-5)**2) * y
    
    # 创建多种阶数的求解器
    orders = [3, 5, 8]
    t_span = [0, 10]
    y0 = 0.0
    
    plt.figure(figsize=(12, 8))
    
    for order in orders:
        solver = TaylorODESolver(variable_stiffness, order=order)
        
        # 使用较松的容差以便观察步长变化
        t, y = solver.solve(t_span, y0, tol=1e-6, max_step=0.1)
        
        # 计算实际使用的步长
        steps = np.diff(t)
        step_points = t[:-1] + steps/2  # 步长的中点
        
        # 绘制解和步长
        plt.subplot(2, 1, 1)
        plt.plot(t, y, '.-', label=f'泰勒{order}阶')
        
        plt.subplot(2, 1, 2)
        plt.semilogy(step_points, steps, '.-', label=f'泰勒{order}阶')
    
    # 添加图例和标签
    plt.subplot(2, 1, 1)
    plt.title('变刚性ODE的数值解')
    plt.xlabel('时间')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.title('自适应步长变化')
    plt.xlabel('时间')
    plt.ylabel('步长 (对数)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/taylor_evaluation/step_size_adaptation.png")
    plt.show()
    
    # 也对比各方法的步长
    methods = {
        "Taylor5": {"solver": TaylorSolver, "params": {"order": 5}},
        "RK45": {"solver": RK45Solver, "params": {}},
        "Radau": {"solver": RadauSolver, "params": {}},
        "Hybrid": {"solver": HybridTaylorSolver, "params": {"order": 5}}
    }
    
    plt.figure(figsize=(12, 8))
    
    for method_name, method_info in methods.items():
        solver = method_info["solver"](variable_stiffness, **method_info["params"])
        
        # 统一误差容限以便比较
        t, y = solver.solve(t_span, y0, tol=1e-6)
        
        # 计算步长
        steps = np.diff(t)
        step_points = t[:-1] + steps/2
        
        # 绘制解和步长
        plt.subplot(2, 1, 1)
        plt.plot(t, y, '.-', label=method_name)
        
        plt.subplot(2, 1, 2)
        plt.semilogy(step_points, steps, '.-', label=method_name)
    
    # 添加图例和标签
    plt.subplot(2, 1, 1)
    plt.title('不同方法的数值解比较')
    plt.xlabel('时间')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.title('不同方法的步长选择比较')
    plt.xlabel('时间')
    plt.ylabel('步长 (对数)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/taylor_evaluation/methods_step_comparison.png")
    plt.show()

def main():
    """运行所有评估"""
    print("开始泰勒展开方法效果评估...\n")
    
    # 运行评估
    evaluate_taylor_order_effects()
    visualize_taylor_expansion_terms()
    compare_taylor_with_other_methods()
    analyze_step_size_adaptation()
    
    print("\n所有评估完成！结果已保存到 results/taylor_evaluation/")

if __name__ == "__main__":
    main()