from abc import ABC, abstractmethod
import numpy as np

class ODESolverBase(ABC):
    """ODE求解器的基类接口"""
    
    @abstractmethod
    def solve(self, t_span, y0, tol=1e-6, **kwargs):
        """求解ODE"""
        pass
    
    @abstractmethod
    def step(self, t, y, h):
        """单步求解"""
        pass
    
    @property
    @abstractmethod
    def name(self):
        """求解器名称"""
        pass

class CompositeODESolver(ODESolverBase):
    """组合式求解器，根据问题类型自动选择最合适的方法"""
    
    def __init__(self, f, solvers=None):
        """
        初始化组合求解器
        
        参数:
            f: 微分方程右侧函数 f(t, y)
            solvers: 可用求解器列表 [求解器1, 求解器2, ...]
        """
        self.f = f
        self.solvers = solvers or []
        self._current_solver = None
    
    def add_solver(self, solver):
        """添加求解器到可用列表"""
        self.solvers.append(solver)
        return self
    
    def solve(self, t_span, y0, tol=1e-6, **kwargs):
        """选择最合适的求解器并求解ODE"""
        # 分析问题特性
        problem_features = self._analyze_problem(f=self.f, t_span=t_span, y0=y0, tol=tol)
        
        # 选择最合适的求解器
        best_solver = self._select_best_solver(problem_features, tol)
        self._current_solver = best_solver
        
        print(f"自动选择求解器: {best_solver.name}")
        return best_solver.solve(t_span, y0, tol=tol, **kwargs)
    
    def step(self, t, y, h):
        """使用当前选定的求解器执行单步"""
        if self._current_solver is None:
            # 默认选择第一个求解器
            self._current_solver = self.solvers[0] if self.solvers else None
            
        if self._current_solver is None:
            raise ValueError("没有可用的求解器")
            
        return self._current_solver.step(t, y, h)
    
    def _analyze_problem(self, f, t_span, y0, tol):
        """分析ODE问题特性"""
        features = {}
        
        # 检查是否是刚性问题
        features['stiffness'] = self._estimate_stiffness(f, t_span[0], y0)
        
        # 检查是否有奇异点
        features['has_singularity'] = self._check_singularity(f, t_span, y0)
        
        # 检查是否是振荡问题
        features['is_oscillatory'] = self._check_oscillation(f, t_span[0], y0)
        
        # 精度要求
        features['precision'] = 'high' if tol < 1e-6 else 'medium' if tol < 1e-3 else 'low'
        
        return features
    
    def _estimate_stiffness(self, f, t, y):
        """估计问题的刚性"""
        try:
            # 简单的刚性估计 - 雅可比矩阵最大特征值与最小特征值之比
            h = 1e-6
            if np.isscalar(y):
                # 标量ODE
                df = (f(t, y + h) - f(t, y)) / h
                return abs(df) > 100  # 简单阈值
            else:
                # 系统ODE
                import numpy.linalg as la
                from scipy.sparse.linalg import eigs
                
                # 计算有限差分雅可比矩阵
                n = len(y)
                jac = np.zeros((n, n))
                f0 = f(t, y)
                
                for i in range(n):
                    y_perturbed = y.copy()
                    y_perturbed[i] += h
                    f1 = f(t, y_perturbed)
                    jac[:, i] = (f1 - f0) / h
                
                # 尝试估计最大和最小特征值
                try:
                    # 对于小矩阵，直接计算所有特征值
                    if n < 10:
                        eigvals = np.linalg.eigvals(jac)
                        max_eig = max(abs(eigvals))
                        min_eig = min(abs(eigvals))
                        return max_eig / min_eig > 1000 if min_eig > 0 else True
                    else:
                        # 对于大矩阵，使用稀疏方法估计最大和最小特征值
                        max_eig = abs(eigs(jac, k=1, which='LM', return_eigenvectors=False)[0])
                        min_eig = abs(eigs(jac, k=1, which='SM', return_eigenvectors=False)[0])
                        return max_eig / min_eig > 1000 if min_eig > 0 else True
                except:
                    # 如果特征值计算失败，假设是刚性的
                    return True
        except:
            # 出错时保守假设可能是刚性的
            return True
    
    def _check_singularity(self, f, t_span, y0):
        """检查ODE是否存在奇异点"""
        try:
            # 在几个点采样函数值，检查是否有极大值或是否有非常不平滑的行为
            t_samples = np.linspace(t_span[0], t_span[1], 10)
            y = y0
            results = []
            
            for t in t_samples:
                try:
                    result = f(t, y)
                    results.append(abs(result) if np.isscalar(result) else np.linalg.norm(result))
                    # 用欧拉法粗略前进
                    h = (t_span[1] - t_span[0]) / 100
                    y = y + h * result
                except:
                    return True  # 出错可能意味着有奇异点
            
            # 检查结果变化是否过大
            if max(results) / (min(results) + 1e-10) > 1e6:
                return True
                
            return False
        except:
            return False
    
    def _check_oscillation(self, f, t, y):
        """检查是否为振荡问题"""
        try:
            if np.isscalar(y):
                # 通过检查f的符号变化来简单判断
                h = 1e-3
                signs = [np.sign(f(t + i*h, y)) for i in range(5)]
                return len(set(signs)) > 1
            else:
                # 对于系统，检查雅可比矩阵是否有纯虚特征值
                n = len(y)
                jac = np.zeros((n, n))
                h = 1e-6
                f0 = f(t, y)
                
                for i in range(n):
                    y_perturbed = y.copy()
                    y_perturbed[i] += h
                    f1 = f(t, y_perturbed)
                    jac[:, i] = (f1 - f0) / h
                
                eigvals = np.linalg.eigvals(jac)
                return any(abs(eigval.real) < 1e-10 and abs(eigval.imag) > 1e-10 for eigval in eigvals)
        except:
            return False
    
    def _select_best_solver(self, features, tol):
        """基于问题特性选择最佳求解器"""
        if not self.solvers:
            raise ValueError("没有可用的求解器")
        
        # 简单的决策逻辑
        if features.get('stiffness', False):
            # 刚性问题优先选择隐式方法
            for solver in self.solvers:
                if 'implicit' in solver.name.lower() or 'bdf' in solver.name.lower():
                    return solver
        
        if features.get('has_singularity', False):
            # 奇异问题优先使用自适应方法
            for solver in self.solvers:
                if 'adaptive' in solver.name.lower():
                    return solver
        
        if features.get('precision', '') == 'high':
            # 高精度优先考虑高阶方法
            for solver in self.solvers:
                if 'taylor' in solver.name.lower() or 'high' in solver.name.lower():
                    return solver
                    
        # 默认使用RK45作为通用可靠的方法
        for solver in self.solvers:
            if 'rk' in solver.name.lower():
                return solver
        
        # 如果没有特别匹配的，返回第一个求解器
        return self.solvers[0]
    
    @property
    def name(self):
        return "CompositeODESolver"