import numpy as np
from numba import jit, vectorize, prange
from scipy.sparse import lil_matrix, csr_matrix
from scipy.special import factorial  # 添加这行导入

try:
    import cupy as cp # type: ignore
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

@jit(nopython=True)
def jit_taylor_step(derivatives, h, order):
    """
    使用Numba JIT加速泰勒展开计算
    
    参数:
        derivatives: 导数列表
        h: 步长
        order: 展开阶数
        
    返回:
        泰勒展开结果
    """
    result = 0.0
    for k in range(order + 1):
        # 使用math.factorial是numba兼容的
        result += derivatives[k] * (h**k) / factorial(k)
    
    # 计算误差
    error = derivatives[order + 1] * (h**(order + 1)) / factorial(order + 1)
    
    return result, np.abs(error)

@jit(nopython=True, parallel=True)
def parallel_taylor_step(derivatives, h, order):
    """
    使用Numba并行加速泰勒展开计算
    """
    result = 0.0
    # 并行计算各阶项
    for k in prange(order + 1):
        result += derivatives[k] * (h**k) / factorial(k)
    
    # 计算误差
    error = derivatives[order + 1] * (h**(order + 1)) / factorial(order + 1)
    
    return result, np.abs(error)

def sparse_jacobian(f, t, y, eps=1e-8):
    """
    使用稀疏矩阵计算雅可比矩阵
    
    参数:
        f: ODE右端函数
        t: 当前时间
        y: 当前状态向量
        eps: 有限差分步长
        
    返回:
        稀疏格式的雅可比矩阵
    """
    n = len(y)
    J = lil_matrix((n, n))
    f0 = f(t, y)
    
    for i in range(n):
        y_perturbed = y.copy()
        y_perturbed[i] += eps
        f1 = f(t, y_perturbed)
        
        # 只存储非零元素
        diff = f1 - f0
        for j in range(n):
            if abs(diff[j]) > eps * 10:  # 过滤掉非常小的值
                J[j, i] = diff[j] / eps
    
    return J.tocsr()  # 转换为CSR格式提高乘法性能

def gpu_taylor_step(derivatives, h, order):
    """
    使用GPU加速泰勒展开计算，如果没有GPU则回退到CPU优化版本
    
    参数:
        derivatives: 导数列表
        h: 步长
        order: 展开阶数
        
    返回:
        泰勒展开结果
    """
    if not HAS_CUPY:
        # 回退到优化的CPU版本
        return intel_optimized_taylor_step(derivatives, h, order)
    
    # GPU计算逻辑 (仅当CUPY可用时执行)
    h_powers = cp.array([h**k / float(factorial(k)) for k in range(order + 1)])
    main_solution = cp.zeros_like(derivatives[0])
    for k in range(order + 1):
        main_solution += derivatives[k] * h_powers[k]
    error_term = derivatives[order + 1] * (h**(order + 1)) / float(factorial(order + 1))
    return main_solution, cp.linalg.norm(error_term)

def intel_optimized_taylor_step(derivatives, h, order):
    """
    为Intel处理器优化的泰勒展开计算
    
    参数:
        derivatives: 导数列表
        h: 步长
        order: 展开阶数
        
    返回:
        泰勒展开结果
    """
    # 预先计算所有h的幂次，利用向量化操作
    h_powers = np.array([h**k / factorial(k) for k in range(order + 1)])
    
    # 使用向量化的点积运算 (Intel MKL会自动优化这部分)
    main_solution = np.dot(derivatives[:order+1], h_powers)
    
    # 计算误差估计项
    error_term = derivatives[order + 1] * (h**(order + 1)) / factorial(order + 1)
    
    return main_solution, np.abs(error_term)

@jit(nopython=True)
def regularized_derivatives(derivatives, factor=0.9):
    """
    正则化导数，减小高阶导数可能引起的数值不稳定性
    
    参数:
        derivatives: 导数列表
        factor: 衰减因子
        
    返回:
        正则化后的导数列表
    """
    reg_derivatives = derivatives.copy()
    for k in range(1, len(derivatives)):
        reg_derivatives[k] *= factor**k
    
    return reg_derivatives

def adaptive_order_selection(h, jacobian_norm, max_order=10):
    """
    根据系统刚性自适应选择泰勒展开阶数
    
    参数:
        h: 当前步长
        jacobian_norm: 雅可比矩阵的范数
        max_order: 最大允许阶数
        
    返回:
        建议的阶数
    """
    # 刚性系统时降低阶数
    stiffness = h * jacobian_norm
    if stiffness < 0.1:
        return max_order
    elif stiffness < 1.0:
        return int(max_order * 0.7)
    elif stiffness < 10.0:
        return int(max_order * 0.5)
    else:
        return int(max_order * 0.3)