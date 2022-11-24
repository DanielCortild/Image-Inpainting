import numpy as np
import scipy as sp
import scipy.linalg

from Image import Image
from Algorithm import Algorithm

class InPainter:
    """
    Solves the inpainting problem on a given image, using an inertial version of the KM 
    iterations, combined with a Bregman iteration.
    Parameters:
        image                 An instance of Image, to be inpainted
        alpha                 Initial value for alpha (Default: 1)
        alpha_static          Whether alpha is static or not (Default: True)
        lamb                  Value of lambda in (0,1) (Default: 0.5)
        rho                   Value of rho in (0,2) (Default: 1)
    Public Methods:
        run                   Runs the algorithm 
    Protected Methods:
    Private Methods:
        prox_f                The proximal operator of the f(Z) = nuclear norm of Z_(1)
        prox_g                The proximal operator of the g(Z) = nuclear norm of Z_(2)
        grad_h                Gradient of the function h(Z) = 1/2 |Z-Z_corrupt|_F^2
    """
    
    def __init__(self, 
                 image: Image, 
                 alpha: float = 1,
                 alpha_static: bool = True,
                 lamb: float = 0.5,
                 rho: float = 1) -> None:

        # Set methods to be used in the Algorithm
        self.__A = image.mask_image
        self.__A_adj = image.mask_image
        self.__getZ1 = image.getZ1
        self.__ungetZ1 = image.ungetZ1
        self.__getZ2 = image.getZ2
        self.__ungetZ2 = image.ungetZ2
        
        # Set parameters to be used in the Algorithm
        self.alpha = alpha
        self.alpha_static = alpha_static
        self.lamb = lamb
        self.rho = rho
        
        # Set the corrupt image
        self.Z_corrupt = image.get_image_masked()
    
    def __prox_f(self, Z: list, rho: float) -> list:
        """ @private
        f(Z) = |Z_(1)|_* (Nuclear Norm of Z_(1))
        If Z_(1) = U @ S @ V^T (SVD Decomposition) then prox_(rho*f)(Z) = U @ S_shrink @ V^T
        """
        U, S, VT = sp.linalg.svd(self.__getZ1(Z), full_matrices=False)
        S_shrink = np.maximum(S - rho, 0)
        return self.__ungetZ1((U * S_shrink) @ VT)

    def __prox_g(self, Z: list, rho: float) -> list:
        """ @private
        g(Z) = |Z_(2)|_* (Nuclear Norm of Z_(2))
        If Z_(2) = U @ S @ V^T (SVD Decomposition) then prox_(rho*g)(Z) = U @ S_shrink @ V^T
        """
        U, S, VT = sp.linalg.svd(self.__getZ2(Z), full_matrices=False)
        S_shrink = np.maximum(S - rho, 0)
        return self.__ungetZ2((U * S_shrink) @ VT)

    def __grad_h(self, Z: list) -> list:
        """ @private
        h(Z) = |Z-Z_corrupt|^2/2
        grad(h)(Z) = Z-Z_corrupt
        """
        return Z - self.Z_corrupt
    
    def __get_alpha(self, k: int) -> float:
        """ @private
        Get the value of alpha at the kth iteration
        """
        return self.alpha if self.alpha_static else (1-1/k) * self.alpha
    
    def __T(self, Y: list) -> list:
        """ @private
        The linear operator T of which we want to find a fixed point
        """
        Yg = self.__prox_g(Y, self.rho)
        return Y - Yg + self.__prox_f(2 * Yg - Y - self.rho * self.__A_adj(self.__grad_h(self.__A(Yg))), self.rho)
    
    def run(self, iterations: int) -> list:
        """ @public
        Run a certain amount of iterations of the Algorithm
        """
        sol = Algorithm( T = self.__T,
                         Z0 = self.Z_corrupt,
                         Z1 = self.Z_corrupt,
                         lamb = self.lamb,
                         get_alpha = self.__get_alpha).run(iterations)
        return sol