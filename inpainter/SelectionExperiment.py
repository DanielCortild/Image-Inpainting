# External Imports
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm.auto import trange
import pandas
from tabulate import tabulate
import multiprocessing

# Internal Imports
from .InPainter import InPainter
from .Image import Image

class SelectionExperiment:
    def __init__(self, var_list: list) -> None:
        # Define the experiment parameters
        self.var_list = var_list
        self.var_name = ""
        self.lamb = 1
        self.rho = 1
        self.ratio = 0.5

    def __run_single(self, i: int, alpha_static: float, max_it: int, tol: float) -> tuple:
        var = self.var_list[i]
        
        if self.var_name == "ratio":
            self.ratio = var
        if self.var_name == "rho":
            self.rho = var
            
        Img = Image(image="Houses.jpeg", ratio = self.ratio, resize=(512, 512))
        start = time.time()
        IP = InPainter(Img, alpha_static = alpha_static, lamb = self.lamb, rho = self.rho)
        its = IP.run(max_it, tol)[1]
        times = time.time() - start
        return its, times
    
    def __run(self, max_it: int, tol: float) -> float:
        its_list_static = np.zeros_like(self.var_list)
        time_list_static = np.zeros_like(self.var_list)
        its_list_inertial = np.zeros_like(self.var_list)
        time_list_inertial = np.zeros_like(self.var_list)
        
        # Run the Static Algorithm for every value of rho
        for i in trange(len(self.var_list), unit=f"Value of {self.var_name}", desc="Static Alpha", leave=False):
            its_list_static[i], time_list_static[i] = self.__run_single(i, True, max_it, tol)
        
        # Run the Ineratial Algorithm for every value of rho
        for i in trange(len(self.var_list), unit=f"Value of {self.var_name}", desc="Inertial Alpha", leave=False):
            its_list_inertial[i], time_list_inertial[i] = self.__run_single(i, False, max_it, tol)
            
        # Return the number of iterations per value of rho
        return its_list_static, its_list_inertial, time_list_static, time_list_inertial
    
    def __plot(self, 
               its_static: list, 
               its_inertial: list, 
               times_static: list,
               times_inertial: list,
               max_iterations: int, 
               tolerance: float,
               title: str = "") -> None:
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
        
    def run(self, max_iterations: int, tolerance: float, title: str = ""):
        its_static, its_inertial, times_static, times_inertial = self.__run(max_iterations, tolerance)
        self.__plot(its_static, its_inertial, times_static, times_inertial, max_iterations, tolerance, title)
        self.__print(its_static, its_inertial, times_static, times_inertial)
        
        
    def __print(self, its_static: list, its_inertial: list, times_static: list, times_inertial: list) -> None:
        data = np.zeros((len(self.var_list), 5))
        data[:, 0] = self.var_list
        data[:, 1] = its_static
        data[:, 2] = times_static
        data[:, 3] = its_inertial
        data[:, 4] = times_inertial
        print(tabulate(data, headers=[self.var_name, "Its Static", "Time Static", "Its Inertial", "Time Inertial"])) 

        
class SelectionExperimentRho ( SelectionExperiment ):
    def __init__(self,
                 var_list: list,
                 lamb: float = 1,
                 ratio: float = 0.5) -> None:
        # Define the experiment parameters
        super().__init__(var_list)
        self.lamb = lamb
        self.ratio = ratio
        
        # Metadata for various plots and methods
        self.var_name = "rho"
        self.legend_loc = 1
        
        
class SelectionExperimentRatio ( SelectionExperiment ):
    def __init__(self,
                 var_list: list,
                 lamb: float = 1,
                 rho: float = 1) -> None:
        # Define the experiment
        super().__init__(var_list)
        self.lamb = lamb
        self.rho = rho
        
        # Metadata for various plots and methods
        self.var_name = "ratio"
        self.legend_loc = 2
        
class SelectionExperimentLambda ( SelectionExperiment ):
    def __init__(self,
                 var_list: list,
                 ratio: float = 1,
                 rho: float = 1) -> None:
        # Define the experiment
        super().__init__(var_list)
        self.ratio = ratio
        self.rho = rho
        
        # Metadata for various plots and methods
        self.var_name = "lambda"
        self.legend_loc = 2