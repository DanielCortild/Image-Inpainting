import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

from Image import Image
from Algorithm import Algorithm

class AlgorithmIKM(Algorithm):
    
    def __init__(self, 
                 image: Image, 
                 alpha: float = 0, 
                 intertial: bool = False,
                 lamb: float = 1, 
                 rho: float = 1) -> None:
        super().__init__(image, "Inertial Krasnoselskii-Mann Iterations")
        
        # Load mask methods from image
        self.__A = image.mask_image
        
        # Create parameters for the iterations
        if intertial: self.get_alpha = lambda k: (1-1/k) * alpha
        else: self.get_alpha = lambda k: alpha
        self.lamb = lamb
        self.rho = rho
                
        # Set initial values
        self.Z0 = self.Z_corrupt
        self.Z1 = self.Z_corrupt
        
    def __prox_f(self, Z: list) -> list:
        """ @private
        f(Z) = |Z_(1)|_* (Nuclear Norm of Z_(1))
        If Z_(1) = U @ S @ V^T (SVD Decomposition) then prox_(rho*f)(Z) = U @ S_shrink @ V^T
        """
        U, S, VT = np.linalg.svd(self._getZ1(Z), full_matrices=False)
        S_shrink = np.maximum(S - self.rho, 0)
        return self._ungetZ1((U * S_shrink) @ VT)

    def __prox_g(self, Z: list) -> list:
        """ @private
        g(Z) = |Z_(2)|_* (Nuclear Norm of Z_(2))
        If Z_(2) = U @ S @ V^T (SVD Decomposition) then prox_(rho*g)(Z) = U @ S_shrink @ V^T
        """
        U, S, VT = np.linalg.svd(self._getZ2(Z), full_matrices=False)
        S_shrink = np.maximum(S - self.rho, 0)
        return self._ungetZ2((U * S_shrink) @ VT)

    def __grad_h(self, Z: list) -> list:
        """ @private
        h(Z) = |Z-Z_corrupt|^2/2
        grad(h)(Z) = Z-Z_corrupt
        """
        return Z - self.Z_corrupt

    def __iterate(self, Z0: list, Z1: list, k: int) -> tuple:
        """ @private
        Perform the iterations according to Algorithm 2
        """
        # Inertial Step
        alpha = self.get_alpha(k)
        U = Z1 + alpha * (Z1 - Z0)
        
        # Krasnoselskii-Mann Step
        XB = self.__prox_g(U)
        XA = self.__prox_f(2 * XB - U - self.rho * self._A_adj(self.__grad_h(self._A(XB))))
        Z2 = U + self.lamb * (XA - XB)
        
        return Z1, Z2
    
    def run(self, iterations: int) -> None:
        """ @public
        Run the algorithm given the number of iterations and the iterator
        """
        for i in tqdm(range(iterations)):
            self.Z0, self.Z1 = self.__iterate(self.Z0, self.Z1, i + 1)
        self.sol = self.Z1