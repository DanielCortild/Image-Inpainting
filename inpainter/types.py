# !/usr/bin/env python
# encoding: utf-8
"""
types.py - File for custom types
~ Daniel Cortild, 16 December 2022
"""

# Standard Imports
from typing import List, TypedDict

# External Imports
import numpy as np


# Dictionary for Historical values
class HistoryDict(TypedDict):
    Z: List[float]
    TZ: List[float]


# Dictionary for Solution data
class SolutionDict(TypedDict):
    solutions: List[np.ndarray]
    rho_values: List[float]
    lamb_values: List[float]
    alpha_static_values: List[bool]
    bregman_values: List[bool]
    titles: List[str]
    iterations: List[int]
    histories: List[HistoryDict]
    times: List[float]
