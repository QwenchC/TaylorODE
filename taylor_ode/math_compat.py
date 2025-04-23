"""兼容符号计算和数值计算的数学函数模块"""
import numpy as np
from sympy import Symbol, sin as sympy_sin, cos as sympy_cos, exp as sympy_exp

def is_symbolic(obj):
    """检查一个对象是否为符号类型"""
    if hasattr(obj, 'is_Symbol') and obj.is_Symbol:
        return True
    if hasattr(obj, 'free_symbols') and len(obj.free_symbols) > 0:
        return True
    return False

def compatible_sin(x):
    """兼容符号和数值计算的sin函数"""
    if is_symbolic(x):
        return sympy_sin(x)
    return np.sin(x)

def compatible_cos(x):
    """兼容符号和数值计算的cos函数"""
    if is_symbolic(x):
        return sympy_cos(x)
    return np.cos(x)

def compatible_exp(x):
    """兼容符号和数值计算的exp函数"""
    if is_symbolic(x):
        return sympy_exp(x)
    return np.exp(x)