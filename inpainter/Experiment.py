#!/usr/bin/env python
# encoding: utf-8
"""
Experiment.py - Implements certain experiments to select the parameters
~ Daniel Cortild, 26 November 2022
"""

# External Imports
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import time
from tqdm.auto import trange #type: ignore
import pandas #type: ignore
from tabulate import tabulate #type: ignore
from typing import List, Tuple

# Internal Imports
from .InPainter import InPainter
from .Image import Image


class Experiment:
    """
    Creates a Experiment which allows to select a specific parameter for the Algorithm described.
    Parameters:
        var_list              List of variables to test against others fixed
    Public Methods:
        run                   Runs the experiment 
    Private Methods:
        run_single            Runs a single time the algorithm
        run                   Runs all the algorithms
        plot                  Plot the number of iterations and the time taken
        print                 Print the number of iterations and the time taken
    """
    
    def __init__(self, var_list: List[float]) -> None:
        # Define the experiment parameters
        self.var_list: List[float] = var_list
        self.var_name: str = ""
        self.lamb: float = 0.5
        self.rho: float = 1
        self.ratio: float = 0.5
        self.legend_loc: int = 1

    def __run_single(self, i: int, alpha_static: bool, max_it: int, tol: float) -> Tuple[int, float]:
        """ @private
        Runs a single time the algorithm
        """
        var: float = self.var_list[i]
        
        if self.var_name == "ratio":
            self.ratio = var
        if self.var_name == "rho":
            self.rho = var
            
        Img: Image = Image(image="Houses.jpeg", ratio = self.ratio, resize = (512, 512))
        start: float = time.time()
        IP: InPainter = InPainter(Img, alpha_static = alpha_static, lamb = self.lamb, rho = self.rho)
        its: int = IP.run(max_it, tol)[1]
        times: float = time.time() - start
        return its, times
    
    def __run(self, max_it: int, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ @private
        Runs all the algorithms
        """
        its_list_static: np.ndarray = np.zeros_like(self.var_list)
        time_list_static: np.ndarray = np.zeros_like(self.var_list)
        its_list_inertial: np.ndarray = np.zeros_like(self.var_list)
        time_list_inertial: np.ndarray = np.zeros_like(self.var_list)
        
        # Run the Static Algorithm for every value of rho
        for i in trange(len(self.var_list), unit=f"Value of {self.var_name}", desc="Static Alpha", leave=False):
            its_list_static[i], time_list_static[i] = self.__run_single(i, True, max_it, tol)
        
        # Run the Ineratial Algorithm for every value of rho
        for i in trange(len(self.var_list), unit=f"Value of {self.var_name}", desc="Inertial Alpha", leave=False):
            its_list_inertial[i], time_list_inertial[i] = self.__run_single(i, False, max_it, tol)
            
        # Return the number of iterations per value of rho
        return its_list_static, its_list_inertial, time_list_static, time_list_inertial
    
    def __plot(self, 
               its_static: np.ndarray, 
               its_inertial: np.ndarray, 
               times_static: np.ndarray,
               times_inertial: np.ndarray,
               max_iterations: int, 
               tolerance: float,
               title: str = "") -> None:
        """ @private
        Plot the number of iterations and the time taken
        """
        # General Figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
        fig.suptitle(title, fontsize=16, y=1.04)
        
        # Number of iterations plot
        axs[0].title.set_text(f"Iterations to reach {tolerance} tolerance")
        axs[0].set_xlabel(f"Value of {self.var_name}")
        axs[0].set_ylabel("Number of iterations")
        axs[0].plot(self.var_list, its_static, label = "Static Iterations", color = "g")
        axs[0].plot(self.var_list, its_inertial, label = "Inertial Iterations", color = "b")
        axs[0].plot(self.var_list[np.argmin(its_static)], np.min(its_static), 'go')
        axs[0].plot(self.var_list[np.argmin(its_inertial)], np.min(its_inertial), 'bo')
        axs[0].axhline(y = max_iterations, color = "r", label = "Did not converge", linestyle = "--")
        axs[0].legend(loc = self.legend_loc)
        
        
        # Time plot
        axs[1].title.set_text(f"Time to reach {tolerance} tolerance")
        axs[1].set_xlabel(f"Value of {self.var_name}")
        axs[1].set_ylabel("Time in seconds")
        axs[1].plot(self.var_list, times_static, label = "Static Iterations", color = "g")
        axs[1].plot(self.var_list, times_inertial, label = "Ineratial Iterations", color = "b")
        axs[1].plot(self.var_list[np.argmin(times_static)], np.min(times_static), 'go')
        axs[1].plot(self.var_list[np.argmin(times_inertial)], np.min(times_inertial), 'bo')
        axs[1].legend(loc = self.legend_loc)
        
        plt.show()
        
    def __print(self, 
                its_static: np.ndarray, 
                its_inertial: np.ndarray, 
                times_static: np.ndarray, 
                times_inertial: np.ndarray) -> None:
        """ @private
        Print the number of iterations and the time taken
        """
        data: np.ndarray = np.zeros((len(self.var_list), 5))
        data[:, 0] = self.var_list
        data[:, 1] = its_static
        data[:, 2] = times_static
        data[:, 3] = its_inertial
        data[:, 4] = times_inertial
        print(tabulate(data, headers=[self.var_name, "Its Static", "Time Static", "Its Inertial", "Time Inertial"])) 
        
    def run(self, max_iterations: int, tolerance: float, title: str = ""):
        """ @public
        Runs the experiment and prints out the resulting data
        """
        its_static, its_inertial, times_static, times_inertial = self.__run(max_iterations, tolerance)
        self.__plot(its_static, its_inertial, times_static, times_inertial, max_iterations, tolerance, title)
        self.__print(its_static, its_inertial, times_static, times_inertial)

        
class ExperimentRho ( Experiment ):
    """
    Tests various values of rho against others fixed.
    Parameters:
        var_list              List of rhos to test against others fixed
        lamb                  Fixed value of lambda
        ratio                 Fixed value of ratio
    Public Methods:
    Protected Methods:
    Private Methods:
    """
        
    def __init__(self,
                 var_list: List[float],
                 lamb: float = 1,
                 ratio: float = 0.5) -> None:
        # Define the experiment parameters
        super().__init__(var_list)
        self.lamb: float = lamb
        self.ratio: float = ratio
        
        # Metadata for various plots and methods
        self.var_name: str = "rho"
        self.legend_loc: int = 1
        
        
class ExperimentRatio ( Experiment ):
    """
    Tests various values of ratio against others fixed.
    Parameters:
        var_list              List of ratios to test against others fixed
        lamb                  Fixed value of lambda
        rho                   Fixed value of rho
    Public Methods:
    Protected Methods:
    Private Methods:
    """
        
    def __init__(self,
                 var_list: List[float],
                 lamb: float = 1,
                 rho: float = 1) -> None:
        # Define the experiment
        super().__init__(var_list)
        self.lamb: float = lamb
        self.rho: float = rho
        
        # Metadata for various plots and methods
        self.var_name: str = "ratio"
        self.legend_loc: int = 2
        
class ExperimentLambda ( Experiment ):
    """
    Tests various values of lambdas against others fixed.
    Parameters:
        var_list              List of lambdas to test against others fixed
        ratio                 Fixed value of ratio
        rho                   Fixed value of rho
    Public Methods:
    Protected Methods:
    Private Methods:
    """
    
    def __init__(self,
                 var_list: List[float],
                 ratio: float = 1,
                 rho: float = 1) -> None:
        # Define the experiment
        super().__init__(var_list)
        self.ratio: float = ratio
        self.rho: float = rho
        
        # Metadata for various plots and methods
        self.var_name: str = "lambda"
        self.legend_loc: int = 2