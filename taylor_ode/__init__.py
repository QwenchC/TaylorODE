# 保留原始类导出
from .core import TaylorODESolver, TaylorSystemSolver

# 添加新的统一接口
from .api import solve_ode
from .base import ODESolverBase, CompositeODESolver
from .solvers import TaylorSolver, RK45Solver, RadauSolver, HybridTaylorSolver
from .utils import analyze_ode, estimate_optimal_order

__version__ = '0.2.0'