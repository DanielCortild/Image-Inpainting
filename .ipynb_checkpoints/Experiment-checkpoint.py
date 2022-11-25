# External Imports
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm.auto import trange
import pandas
from tabulate import tabulate
import multiprocessing

# Internal Imports
from InPainter import InPainter
from Image import Image

class ExperimentRho:
    def __init__(self,
                 rho_list: list = None,
                 lamb: float = 1,
                 ratio: float = 0.5) -> None:
        # Define the standard parameters
        self.image = Image(image="Houses.jpeg", ratio = ratio, resize=(512, 512))
        
        # Define the experiment
        self.rho_list = rho_list
        self.lamb = lamb
            
    def __run_single(self, i: int, alpha_static: float, max_it: int, tol: float) -> tuple:
        rho = self.rho_list[i]
        start = time.time()
        IP = InPainter(self.image, alpha_static = alpha_static, lamb = self.lamb, rho = rho)
        its = IP.run(max_it, tol)[1]
        times = time.time() - start
        return its, times
            
    def __run(self, max_it: int, tol: float) -> float:
        its_list_static = np.zeros_like(self.rho_list)
        time_list_static = np.zeros_like(self.rho_list)
        its_list_inertial = np.zeros_like(self.rho_list)
        time_list_inertial = np.zeros_like(self.rho_list)
        
        # Run the Static Algorithm for every value of rho
        for i in trange(len(self.rho_list), unit="Value of rho", desc="Static Alpha", leave=False):
            its_list_static[i], time_list_static[i] = self.__run_single(i, True, max_it, tol)
        
        # Run the Ineratial Algorithm for every value of rho
        for i in trange(len(self.rho_list), unit="Value of rho", desc="Inertial Alpha", leave=False):
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
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(title)
        
        # Number of iterations plot
        axs[0].title.set_text(f"Iterations to reach {tolerance} tolerance")
        axs[0].set_xlabel("Value of rho")
        axs[0].set_ylabel("Number of iterations")
        axs[0].plot(self.rho_list, its_static, label = "Static Iterations", color = "g")
        axs[0].plot(self.rho_list, its_inertial, label = "Inertial Iterations", color = "b")
        axs[0].plot(self.rho_list[np.argmin(its_static)], np.min(its_static), 'go')
        axs[0].plot(self.rho_list[np.argmin(its_inertial)], np.min(its_inertial), 'bo')
        axs[0].axhline(y = max_iterations, color = "r", label = "Did not converge", linestyle = "--")
        axs[0].legend()
        
        
        # Time plot
        axs[1].title.set_text(f"Time to reach {tolerance} tolerance")
        axs[1].set_xlabel("Value of rho")
        axs[1].set_ylabel("Time in seconds")
        axs[1].plot(self.rho_list, times_static, label = "Static Iterations", color = "g")
        axs[1].plot(self.rho_list, times_inertial, label = "Ineratial Iterations", color = "b")
        axs[1].plot(self.rho_list[np.argmin(times_static)], np.min(times_static), 'go')
        axs[1].plot(self.rho_list[np.argmin(times_inertial)], np.min(times_inertial), 'bo')
        axs[1].legend()
        
        plt.show()
        
    def __print(self, its_static: list, its_inertial: list, times_static: list, times_inertial: list) -> None:
        data = np.zeros((len(self.rho_list), 5))
        data[:, 0] = self.rho_list
        data[:, 1] = its_static
        data[:, 2] = times_static
        data[:, 3] = its_inertial
        data[:, 4] = times_inertial
        print(tabulate(data, headers=["Rho", "Its Static", "Time Static", "Its Inertial", "Time Inertial"]))
    
    def run(self, max_iterations: int, tolerance: float, title: str = ""):
        its_static, its_inertial, times_static, times_inertial = self.__run(max_iterations, tolerance)
        self.__plot(its_static, its_inertial, times_static, times_inertial, max_iterations, tolerance, title)
        self.__print(its_static, its_inertial, times_static, times_inertial)
        
class ExperimentPerc:
    def __init__(self,
                 perc_list: list = None,
                 lamb: float = 1,
                 rho: float = 1) -> None:
        # Define the experiment
        self.perc_list = perc_list
        self.lamb = lamb
        self.rho = rho
            
    def __run_single(self, i: int, alpha_static: float, max_it: int, tol: float) -> tuple:
        perc = self.perc_list[i]
        Img = Image(image="Houses.jpeg", ratio = perc, resize=(512, 512))
        start = time.time()
        IP = InPainter(Img, alpha_static = alpha_static, lamb = self.lamb, rho = self.rho)
        its = IP.run(max_it, tol)[1]
        times = time.time() - start
        return its, times
            
    def __run(self, max_it: int, tol: float) -> float:
        its_list_static = np.zeros_like(self.perc_list)
        time_list_static = np.zeros_like(self.perc_list)
        its_list_inertial = np.zeros_like(self.perc_list)
        time_list_inertial = np.zeros_like(self.perc_list)
        
        # Run the Static Algorithm for every value of perc
        for i in trange(len(self.perc_list), unit="Value of rho", desc="Static Alpha", leave=False):
            its_list_static[i], time_list_static[i] = self.__run_single(i, True, max_it, tol)
        
        # Run the Ineratial Algorithm for every value of perc
        for i in trange(len(self.perc_list), unit="Value of rho", desc="Inertial Alpha", leave=False):
            its_list_inertial[i], time_list_inertial[i] = self.__run_single(i, False, max_it, tol)
            
        # Return the number of iterations per value of perc
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
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(title)
        
        # Number of iterations plot
        axs[0].title.set_text(f"Iterations to reach {tolerance} tolerance")
        axs[0].set_xlabel("Percentage of erased pixels")
        axs[0].set_ylabel("Number of iterations")
        axs[0].plot(self.perc_list, its_static, label = "Static Iterations", color = "g")
        axs[0].plot(self.perc_list, its_inertial, label = "Inertial Iterations", color = "b")
        axs[0].plot(self.perc_list[np.argmin(its_static)], np.min(its_static), 'go')
        axs[0].plot(self.perc_list[np.argmin(its_inertial)], np.min(its_inertial), 'bo')
        axs[0].axhline(y = max_iterations, color = "r", label = "Did not converge", linestyle = "--")
        axs[0].legend()
        
        
        # Time plot
        axs[1].title.set_text(f"Time to reach {tolerance} tolerance")
        axs[1].set_xlabel("Percentage of erased pixels")
        axs[1].set_ylabel("Time in seconds")
        axs[1].plot(self.perc_list, times_static, label = "Static Iterations", color = "g")
        axs[1].plot(self.perc_list, times_inertial, label = "Ineratial Iterations", color = "b")
        axs[1].plot(self.perc_list[np.argmin(times_static)], np.min(times_static), 'go')
        axs[1].plot(self.perc_list[np.argmin(times_inertial)], np.min(times_inertial), 'bo')
        axs[1].legend()
        
        plt.show()
        
    def __print(self, its_static: list, its_inertial: list, times_static: list, times_inertial: list) -> None:
        data = np.zeros((len(self.perc_list), 5))
        data[:, 0] = self.perc_list
        data[:, 1] = its_static
        data[:, 2] = times_static
        data[:, 3] = its_inertial
        data[:, 4] = times_inertial
        print(tabulate(data, headers=["Percentage", "Its Static", "Time Static", "Its Inertial", "Time Inertial"]))
    
    def run(self, max_iterations: int, tolerance: float, title: str = ""):
        its_static, its_inertial, times_static, times_inertial = self.__run(max_iterations, tolerance)
        self.__plot(its_static, its_inertial, times_static, times_inertial, max_iterations, tolerance, title)
        self.__print(its_static, its_inertial, times_static, times_inertial)