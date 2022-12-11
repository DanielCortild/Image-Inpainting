#!/usr/bin/env python
# encoding: utf-8
"""
convergence.py - Implements a function that plots several convergence plots
~ Daniel Cortild, 26 November 2022
"""


# External Imports
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from typing import List


def plot_convergence(hist_static: List[np.ndarray], hist_inertial: List[np.ndarray]) -> None:
    """
    Create plots for convergence analysis
    """
    # Compute the convergence rates
    conv_static = [np.linalg.norm(z - Tz)**2 for z, Tz in zip(hist_static[0], hist_static[1])]
    conv_inertial = [np.linalg.norm(z - Tz)**2 for z, Tz in zip(hist_inertial[0], hist_inertial[1])]

    # Plot Converge of Solutions
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    fig.suptitle("Convergence of Solutions", fontsize=16, y=1.04)

    # First Plot
    axs[0].title.set_text("Convergence of $|z_k-Tz_k|^2$")
    axs[0].set_xlabel("Iteration ($k$)")
    axs[0].set_ylabel(f"$|z_k-Tz_k|^2$")
    axs[0].set_yscale('log')
    axs[0].plot(np.arange(len(conv_static))+1, conv_static, label="Static Iterations", color="g")
    axs[0].plot(np.arange(len(conv_inertial))+1, conv_inertial, label="Inertial Iterations", color="b")
    axs[0].legend()

    # Second Plot
    axs[1].title.set_text("Convergence of $k|z_k-Tz_k|^2$")
    axs[1].set_xlabel("Iteration ($k$)")
    axs[1].set_ylabel(f"$k|z_k-Tz_k|^2$")
    axs[1].set_yscale('log')
    axs[1].plot(np.arange(len(conv_static))+1, np.multiply(np.arange(len(conv_static))+1, conv_static), 
                label="Static Iterations", color="g")
    axs[1].plot(np.arange(len(conv_inertial))+1, np.multiply(np.arange(len(conv_inertial))+1, conv_inertial), 
                label="Inertial Iterations", color="b")
    axs[1].legend()

    plt.show()
