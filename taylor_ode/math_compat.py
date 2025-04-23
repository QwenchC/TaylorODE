"""兼容符号计算和数值计算的数学函数模块"""
import numpy as np
from sympy import symbols, sin as sympy_sin, cos as sympy_cos, exp as sympy_exp

def compatible_sin(t):
    """兼容符号和数值计算的sin函数"""
    if hasattr(t, 'is_Symbol') or hasattr(t, 'free_symbols'):
        return sympy_sin(t)
    return np.sin(t)

def compatible_cos(t):
    """兼容符号和数值计算的cos函数"""
    if hasattr(t, 'is_Symbol') or hasattr(t, 'free_symbols'):
        return sympy_cos(t)
    return np.cos(t)

def compatible_exp(t):
    """兼容符号和数值计算的exp函数"""
    if hasattr(t, 'is_Symbol') or hasattr(t, 'free_symbols'):
        return sympy_exp(t)
    return np.exp(t)