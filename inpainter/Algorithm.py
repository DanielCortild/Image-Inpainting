#!/usr/bin/env python
# encoding: utf-8
"""
Algorithm.py - Implements an Algorithm based on Inertial Krasnoselskii-Mann Iterations
~ Daniel Cortild, 26 November 2022
"""

# External imports
import numpy as np
from tqdm.auto import trange              # type: ignore
from typing import Tuple, Callable, List

# Internal Imports
from .Image import Image

class Algorithm:
    """
    Solves problems of the type: Find p^* such that Fix(T)={p^*}
    Using the algorithm described in Inertial Krasnoselskii-Mann Iterations (Ignacio Fierro, 
    Juan José Maulén, Juan Peypouquet), namely by applying an Inertia Step followed by a 
    Krasnoselskii-Mann Iteration. 
    Parameters:
        T                     Linear Operator T of which we aim to find the fixed point
                              We require T to be q-quasi-contractive
        Z0                    Initial guess for Z0
        Z1                    Initial guess for Z1
        lamb                  Value of lambda in (0,1)
        get_alpha             Nondecreasing sequence in [0,1) getting the value of alpha at iteration k
    Public Methods:
        run                   Runs the algorithm 
    Private Methods:
        iterate               Runs a single iteration of the algorithm
        R                     Error estimate used for stopping criterion
    """
    
    
    def __init__(self, 
                 T: Callable[[np.ndarray], np.ndarray],
                 Z0: np.ndarray,
                 Z1: np.ndarray,
                 lamb: float,
                 get_alpha: Callable[[int], float]) -> None:
        
        # Store the given functions
        self.__T: Callable[[np.ndarray], np.ndarray] = T
        
        # Create parameters for the iterations
        self.__get_alpha: Callable[[int], float] = get_alpha
        self.__lamb: float = lamb
                
        # Set initial values
        self.__Z0: np.ndarray = Z0
        self.__Z1: np.ndarray = Z1
        
    def __iterate(self, Z0: np.ndarray, Z1: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """ @private
        Perform the iterations according to Algorithm 2
        """
        # Inertial Step
        Y: np.ndarray = Z1 + self.__get_alpha(k) * (Z1 - Z0)
        
        # Krasnoselskii-Mann Step
        Z2: np.ndarray = (1 - self.__lamb) * Y + self.__lamb * self.__T(Y)
        
        # Bregman Iteration
        # Yet to be implemented
        
        return Z1, Z2
    
    @staticmethod
    def __R(Z0: np.ndarray, Z1: np.ndarray) -> float:
        """ @private
        Error estimate used for stopping criterion
        """
        return np.linalg.norm(Z1 - Z0) / np.linalg.norm(Z0)
    
    def run(self, max_iterations: int, tolerance: float, verbose: bool = False) -> Tuple[np.ndarray, int, List[float]]:
        """ @public
        Run the algorithm given the number of iterations and the iterator
        """
        Z0, Z1 = self.__Z0, self.__Z1
        Z_conv_hist = []
        for i in trange(max_iterations, disable=not verbose):
            Z0, Z1 = self.__iterate(Z0, Z1, i)
            Z_conv_hist.append(np.linalg.norm(Z1 - self.__T(Z1)) ** 2)
            if self.__R(Z0, Z1) < tolerance:
                break
        return Z1, i + 1, Z_conv_hist