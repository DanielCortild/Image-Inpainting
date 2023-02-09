#!/usr/bin/env python
# encoding: utf-8
"""
InPainter.py - Implements the InPainter Class, which solves the InPainting problem 
               according to certain parameters
~ Daniel Cortild, 15 December 2022
"""

# Standard Imports
import warnings
from time import time
from typing import Callable, Tuple

# External Imports
import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

# Internal Imports
from .types import HistoryDict
from .Image import MaskedImage
from .Algorithm import Algorithm

plt.rcParams.update({'axes.facecolor': 'white'})


class InPainter:
    """
    Solves the inpainting problem on a given image, using an inertial version of the KM
    iterations, combined with a Bregman update.
    Parameters:
        image                 An instance of Image, to be inpainted
        alpha_static          Whether alpha is static or not (Default: True)
        lamb                  Value of lambda in (0,1) (Default: 0.5)
        rho                   Value of rho in (0,2) (Default: 1)
    Public Methods:
        run                   Runs the algorithm
    Private Methods:
    """

    def __init__(self,
                 image: MaskedImage,
                 max_it: int = 100,
                 tol: float = 1e-3,
                 tol_bregman: float = 5e-2,
                 verbose: bool = False) -> None:
        # Save parameters
        self.__image = image
        self.__max_it: int = max_it
        self.__tol: float = tol
        self.__tol_bregman: float = tol_bregman
        self.__verbose: bool = verbose

        # Set methods to be used in the Algorithm
        self.__A: Callable[[np.ndarray], np.ndarray] = image.mask_image

        # Set the corrupt image
        self.__Z_corrupt: np.ndarray = image.get_image_masked()
        
    def run(self, rho: float, lamb: float, alpha_static: bool, bregman: bool = False) \
            -> Tuple[np.ndarray, int, HistoryDict]:
        """ @public
        Run a certain amount of iterations of the Algorithm
        """
        self.__Z_corrupt_copy: np.ndarray = self.__Z_corrupt.copy()

        def bregman_update(Z: np.ndarray) -> None:
            self.__Z_corrupt_copy += rho * (self.__Z_corrupt - self.__A(Z))

        algo = Algorithm(proxf=lambda Z, r: unfold(svd_shrink(fold(Z, axis=0), r), axis=0),
                         proxg=lambda Z, r: unfold(svd_shrink(fold(Z, axis=1), r), axis=1),
                         LgradhL=lambda Z: self.__A(Z - self.__Z_corrupt_copy),
                         update_LgradhL=bregman_update,
                         Z_init=self.__Z_corrupt,
                         lamb=lamb,
                         rho=rho,
                         beta=1,
                         alpha_static=alpha_static)

        start = time()

        iterations, history = algo.run(self.__max_it, self.__tol, self.__tol_bregman if bregman else 0, self.__verbose)

        return history["TZ"][-1], iterations, time() - start, history
    
#     def show(self, title: str = "") -> None:
#         """ @public
#         Visualize the results by plotting the following three images:
#         - The original image, technically inaccessible
#         - The in-painted image, technically the only one accessible
#         - The corrected image
#         """
#         fig = plt.figure(dpi=600)
#         fig.set_figwidth(12)
#         fig.suptitle(title, fontsize=18)

#         ax = fig.add_subplot(1, len(self.__solution["solutions"]) + 2, 1)
#         ax.title.set_text("Original Image")
#         ax.imshow(self.__image.get_image())
#         ax.set_axis_off()

#         ax = fig.add_subplot(1, len(self.__solution["solutions"]) + 2, 2)
#         ax.title.set_text(f"In-Painted Image ({self.__image.get_erase_ratio() * 100} %)")
#         ax.imshow(self.__image.get_image_masked())
#         ax.set_axis_off()

#         for index, sol in enumerate(self.__solution["solutions"]):
#             ax = fig.add_subplot(1, len(self.__solution["solutions"]) + 2, index + 3)
#             ax.title.set_text(self.__solution["titles"][index])
#             ax.imshow(sol)
#             ax.set_axis_off()

#         fig.tight_layout()
#         plt.show()

#     def __show_convergence(self, title: str, var_name: str, var_index: str) -> None:
#         """ @private
#         Displays a convergence plot
#         """
#         # Initialise empty lists to be filled with appropriate values
#         var_list_static = []
#         var_list_inertial = []
#         its_static = []
#         its_inertial = []
#         times_static = []
#         times_inertial = []

#         # Fill empty lists with corresponding values
#         for i, var in enumerate(self.__solution[var_index]):
#             if self.__solution["alpha_static_values"][i]:
#                 var_list_static.append(var)
#                 its_static.append(self.__solution["iterations"][i])
#                 times_static.append(self.__solution["times"][i])
#             else:
#                 var_list_inertial.append(var)
#                 its_inertial.append(self.__solution["iterations"][i])
#                 times_inertial.append(self.__solution["times"][i])

#         # General Figure
#         fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
#         fig.suptitle(title, fontsize=16, y=1.04)

#         # Number of iterations plot
#         axs[0].title.set_text(f"Iterations to reach {self.__tol} tolerance")
#         axs[0].set_xlabel(f"Value of {var_name}")
#         axs[0].set_ylabel("Number of iterations")
#         axs[0].plot(var_list_static, its_static, label="Static Iterations", color="g")
#         axs[0].plot(var_list_inertial, its_inertial, label="Inertial Iterations", color="b")
#         axs[0].axhline(y=self.__max_it, color="r", label="Did not converge", linestyle="--")
#         axs[0].legend()

#         # Time plot
#         axs[1].title.set_text(f"Time to reach {self.__tol} tolerance")
#         axs[1].set_xlabel(f"Value of {var_name}")
#         axs[1].set_ylabel("Time in seconds")
#         axs[1].plot(var_list_static, times_static, label="Static Iterations", color="g")
#         axs[1].plot(var_list_inertial, times_inertial, label="Inertial Iterations", color="b")
#         axs[1].legend()

#         plt.show()

#     def show_rho(self, title: str = "") -> None:
#         """ @public
#         Displays plots of value of rho vs iterations and value of rho vs time
#         """
#         self.__show_convergence(title, var_name="rho", var_index="rho_values")


def fold(Z: np.ndarray, axis: int) -> np.ndarray:
    """
    Transforms a (N, M, 3) tensor to a (N, 3*M) or (3*N, M) tensor
    Reverses un fold
    """
    (a, b) = (1, 3) if axis == 0 else (3, 1)
    return Z.reshape(Z.shape[0] * a, Z.shape[1] * b, order='F')


def unfold(Z: np.ndarray, axis: int) -> np.ndarray:
    """
    Transforms a (N, 3*M) or (3*N, M) tensor to a (N, M, 3) tensor
    Reverses fold
    """
    (a, b) = (1, 3) if axis == 0 else (3, 1)
    return Z.reshape(Z.shape[0] // a, Z.shape[1] // b, 3, order='F')


def svd_shrink(matrix: np.ndarray, rho: float) -> np.ndarray:
    """
    Computes the shrunken SVD of a matrix
    """
    U: np.ndarray
    S: np.ndarray
    VT: np.ndarray
    U, S, VT = sp.linalg.svd(matrix, full_matrices=False)
    return (U * np.maximum(S - rho, 0)) @ VT
