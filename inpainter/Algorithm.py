#!/usr/bin/env python
# encoding: utf-8
"""
Algorithm.py - Implements an Algorithm based on Inertial Krasnoselskii-Mann Iterations
~ Daniel Cortild, 26 November 2022
"""

# External imports
import numpy as np
from tqdm.auto import trange

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
    Protected Methods:
    Private Methods:
        iterate               Runs a single iteration of the algorithm
        R                     Error estimate used for stopping criterion
    """
    
    def __init__(self, 
                 T: callable,
                 Z0: list,
                 Z1: list,
                 lamb: float,
                 get_alpha: callable) -> None:
        
        # Store the given functions
        self.__T = T
        
        # Create parameters for the iterations
        self.__get_alpha = get_alpha
        self.lamb = lamb
                
        # Set initial values
        self.Z0 = Z0
        self.Z1 = Z1
        
    def __iterate(self, Z0: list, Z1: list, k: int) -> tuple:
        """ @private
        Perform the iterations according to Algorithm 2
        """
        # Inertial Step
        Y = Z1 + self.__get_alpha(k) * (Z1 - Z0)
        
        # Krasnoselskii-Mann Step
        Z2 = (1 - self.lamb) * Y + self.lamb * self.__T(Y)
        
        # Bregman Iteration
        # Yet to be implemented
        
        return Z1, Z2
    
    def __R(self, Z0: list, Z1: list) -> float:
        """ @private
        Error estimate used for stopping criterion
        """
        return np.linalg.norm(Z1 - Z0) / np.linalg.norm(Z0)
    
    def run(self, max_iterations: int, tolerance: float, verbose: bool = False) -> list:
        """ @public
        Run the algorithm given the number of iterations and the iterator
        """
        Z0, Z1 = self.Z0, self.Z1
        Z_conv_hist = []
        for i in trange(max_iterations, disable=not verbose):
            Z0, Z1 = self.__iterate(Z0, Z1, i)
            Z_conv_hist.append(np.linalg.norm(Z1 - self.__T(Z1)) ** 2)
            if self.__R(Z0, Z1) < tolerance:
                break
        return Z1, i + 1, Z_conv_hist