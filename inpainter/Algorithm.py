# !/usr/bin/env python
# encoding: utf-8
"""
Algorithm2.py - Implements an Algorithm based on Inertial Krasnoselskii-Mann Iterations
~ Daniel Cortild, 11 December 2022
"""

# Standard Imports
from typing import Tuple, Callable, List, TypedDict

# External imports
import numpy as np
from tqdm.auto import trange                # type: ignore

# Internal Imports
from .types import HistoryDict


class Algorithm:
    """
    Solves minimisation problems over a Hilbert space H of the type:
            min_{x in H} f(x) + g(x) + h(Lx)
    Where f and g are convex lower-semicontinuous, h is convex with 1/beta-Lipschitz continuous gradient
    and L is a bounded linear operator on H. In order to solve this problem, an inertial
    KM iteration is applied. In other words, the algorithm is as follows
        Initialisation: Choose Z0, Z1 in H, lambda a value in (0,1), rho in (0,2), epsilon>0, and
            alpha according to alpha(1+alpha)+(1+1/lambda)alpha(1-alpha)+(1-1/lambda)(1-alpha)=0
            alpha(k) is given by (1-1/k)*alpha in the inertial case and 0 in the static case
        Algorithm:
            while residual(Z1, Z0) > epsilon do
                U = Z1 + alpha(k) * (Z1 - Z0)
                Xg = prox_{rho g}(U)
                Zhalf = 2 * Xg - U - rho * L^*(grad_h(L(Xg)))
                Z0 = Z1
                Z1 = U + lambda * (prox_{rho f}(Zhalf) - Xg)
            end
    Parameters:
        proxf               The proximal operator of f
        proxg               The proximal operator of g
        LgradhL             The operator L^*(grad_h(L))
        Z_init              Initial guess of Z
        lamb                Value of lambda in (0,1) [Default: 0.5]
        rho                 Value of rho in (0,2) [Default: 1]
        beta                The inverse of the Lipschitz constant of grad_h [Default: 1]
        alpha_static        Boolean expression whether alpha is static or not [Default: False]
    Public Methods:
        run                Runs the algorithm
    Private Methods:
        iterate            Runs a single iteration of the algorithm
        error              Compute error estimate used for stopping criterion
    """

    def __init__(self,
                 proxf: Callable[[np.ndarray, float], np.ndarray],
                 proxg: Callable[[np.ndarray, float], np.ndarray],
                 LgradhL: Callable[[np.ndarray], np.ndarray],
                 update_LgradhL: Callable[[None], None],
                 Z_init: np.ndarray,
                 lamb: float = 0.5,
                 rho: float = 1,
                 beta: float = 1,
                 alpha_static: bool = False) -> None:
        self.__proxf: Callable[[np.ndarray, float], np.ndarray] = proxf
        self.__proxg: Callable[[np.ndarray, float], np.ndarray] = proxg
        self.__LgradhL: Callable[[np.ndarray], np.ndarray] = LgradhL
        self.__update_LgradhL: Callable[[None], None] = update_LgradhL
        self.__Z_init: np.ndarray = Z_init
        self.__lambda: float = lamb
        self.__rho: float = rho

        # Compute get_alpha
        eta: float = lamb * 2 * beta / (4 * beta - rho)
        alpha: float = 1 / 3
        if abs(eta - 1 / 2) > 1e-6:
            alpha = (eta - 2 + np.sqrt((eta - 2) ** 2 - 4 * (eta - 1) * (2 * eta - 1))) / (2 * (2 * eta - 1))
        self.__get_alpha: Callable[[int], float] = lambda k: (1 - 1 / (k+1)) * alpha if not alpha_static else 0

    def __operator_T(self, U: np.ndarray) -> np.ndarray:
        """ @private
        Applies the operator T on the inertial variable U
        """
        Xg = self.__proxg(U, self.__rho)
        Z_halfnext = 2 * Xg - U - self.__rho * self.__LgradhL(Xg)
        Z_next = U + self.__lambda * (self.__proxf(Z_halfnext, self.__rho) - Xg)
        return Z_next

    def __iterate(self, Z_previous: np.ndarray, Z_actual: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """ @private
        Perform the iterations according to Algorithm 2
        """
        U = Z_actual + self.__get_alpha(k) * (Z_actual - Z_previous)    # Inertial Step
        Z_next = self.__operator_T(U)                                   # KM Step
        return Z_actual, Z_next

    @staticmethod
    def __residual(Z_previous: np.ndarray, Z_actual: np.ndarray) -> float:
        """ @private
        Computes the residual of the iterations
        """
        return np.linalg.norm(Z_actual - Z_previous) / np.linalg.norm(Z_previous)

    @staticmethod
    def __residual_bregman(Z_previous: np.ndarray, Z_actual: np.ndarray) -> float:
        """ @private
        Computes the residual of the iterations
        """
        return np.linalg.norm(Z_actual - Z_previous)

    def run(self, max_it: int, tol: float, tol_bregman: float = 0, verbose: bool = True) \
            -> Tuple[int, HistoryDict]:
        """ @public
        Run the algorithm given the number of iterations and the iterator
        """
        Z_previous: np.ndarray = np.zeros_like(self.__Z_init) + 1
        Z_next: np.ndarray = self.__Z_init
        hist: HistoryDict = {"Z": [], "TZ": []}
        its: int = 0

        for its in trange(max_it, disable=not verbose):
            Z_previous, Z_next = self.__iterate(Z_previous, Z_next, its)
            hist["Z"].append(Z_next)
            hist["TZ"].append(self.__operator_T(Z_next))
            if self.__residual_bregman(Z_previous, Z_next) < tol_bregman:
                self.__update_LgradhL(Z_next)
            if self.__residual(Z_previous, Z_next) < tol:
                break

        return its + 1, hist
