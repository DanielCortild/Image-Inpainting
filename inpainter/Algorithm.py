# !/usr/bin/env python
# encoding: utf-8
"""
Algorithm.py - Implements an Algorithm based on Inertial Krasnoselskii-Mann Iterations
~ Daniel Cortild, 11 December 2022
"""

# Standard Imports
from typing import Tuple, Callable, List

# External imports
import numpy as np
from tqdm.auto import trange                # type: ignore


class Algorithm:
    """
    Solves problems of the type: Find p^* such that Fix(T)={p^*}
    Using the algorithm described in Inertial Krasnoselskii-Mann Iterations (Ignacio Fierro,
    Juan José Maulén, Juan Peypouquet), namely by applying an Inertia Step followed by a
    Krasnoselskii-Mann Iteration.
    Parameters:
        operator           Linear Operator T of which we aim to find the fixed point
                           We require T to be q-quasi-contractive
        Z_init             Pair (Z0, Z1) of initial values
        lamb               Value of lambda in (0,1)
        get_alpha          Nondecreasing sequence in [0,1) getting the value of alpha at iteration k
    Public Methods:
        run                Runs the algorithm
    Private Methods:
        iterate            Runs a single iteration of the algorithm
        error              Error estimate used for stopping criterion
    """

    def __init__(self,
                 operator: Callable[[np.ndarray], np.ndarray],
                 Z_init: Tuple[np.ndarray, np.ndarray],
                 lamb: float,
                 get_alpha: Callable[[int], float]) -> None:
        self.__operator: Callable[[np.ndarray], np.ndarray] = operator
        self.__get_alpha: Callable[[int], float] = get_alpha
        self.__lamb: float = lamb
        self.__Z_init: np.ndarray = Z_init

    def __iterate(self, Z0: np.ndarray, Z1: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """ @private
        Perform the iterations according to Algorithm 2
        """
        Y: np.ndarray = Z1 + self.__get_alpha(k) * (Z1 - Z0)                        # Inertial Step
        Z2: np.ndarray = (1 - self.__lamb) * Y + self.__lamb * self.__operator(Y)   # KM Step
        return Z1, Z2

    @staticmethod
    def __error(Z0: np.ndarray, Z1: np.ndarray) -> float:
        """ @private
        Error estimate used for stopping criterion
        """
        return np.linalg.norm(Z1 - Z0) / np.linalg.norm(Z0)

    def run(self, max_it: int, tol: float) -> Tuple[np.ndarray, int, List[List[float]]]:
        """ @public
        Run the algorithm given the number of iterations and the iterator
        """
        Z0: np.ndarray = self.__Z_init[0]
        Z1: np.ndarray = self.__Z_init[1]
        hist: List[float] = []
        hist_operator: List[float] = []
        i = 0
        while i <= max_it and self.__error(Z0, Z1) >= tol:
            Z0, Z1 = self.__iterate(Z0, Z1, i)
            hist.append(Z1)
            hist_operator.append(self.__operator(Z1))
        return Z1, i + 1, [hist, hist_operator]
