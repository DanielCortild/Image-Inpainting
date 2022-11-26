#!/usr/bin/env python
# encoding: utf-8
"""
__init__.py - __init__ file for the inpainter module
~ Daniel Cortild, 26 November 2022
"""

from .Image import Image
from .Algorithm import Algorithm
from .InPainter import InPainter
from .Experiment import ExperimentRho as ExpRho, \
                        ExperimentRatio as ExpRatio, \
                        ExperimentLambda as ExpLambda
from .convergence import plot_convergence