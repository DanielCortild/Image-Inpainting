import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from Image import Image

class Algorithm:
    def __init__(self, image: Image, name: str) -> None:
        """
        Initial Values for Algorithm
        """
        self.image = image
        self.name = name
        
        self.Z_corrupt = self.image._get_image_masked()
        self.sol = np.zeros_like(self.Z_corrupt)
    
    def _A(self, Z: list) -> list:
        """ @protected
        The linear operator A, selecting the pixels we know to be correct
        """
        if len(Z.shape) == 3:
            AZ = np.zeros_like(Z)
            mask = self.image._get_mask()
            k = 0
            for i, j in np.ndindex(AZ.shape[:2]):
                if mask[i,j] == 1:
                    AZ[i, j, :] = Z[i, j, :]
                    k += 1
            return AZ
        elif len(Z.shape) == 2:
            AZ = np.zeros_like(Z)
            mask = self.image._get_mask()
            k = 0
            for i, j in np.ndindex(AZ.shape[:2]):
                if mask[i,j] == 1:
                    AZ[i, j] = Z[i, j]
                    k += 1
            return AZ

    def _A_adj(self, AZ: list) -> list:
        """ @protected
        The adjoint of the operator A, the zero-upsampling operator
        """
        return self._A(AZ)
    
    def _getZ1(self, Z: list) -> list:
        """ @protected
        Transforms a (N, M, 3) tensor to a (N, 3*M) tensor
        Reverses _ungetZ1
        """
        return np.hstack([np.array(Z[:, :, 0]), np.array(Z[:, :, 1]), np.array(Z[:, :, 2])])

    def _ungetZ1(self, Z1: list) -> list:
        """ @protected
        Transforms a (N, 3*M) tensor to a (N, M, 3) tensor
        Reverses _getZ1
        """
        N, M = Z1.shape
        M //= 3
        
        Z = np.zeros((N, M, 3))
        Z[:, :, 0] = Z1[:, :M]
        Z[:, :, 1] = Z1[:, M:2*M]
        Z[:, :, 2] = Z1[:, 2*M:]
        return Z
    
    def _getZ2(self, Z: list) -> list:
        """ @protected
        Transforms a (N, M, 3) tensor to a (3*N, M) tensor
        Reverses _ungetZ2
        """
        return np.vstack([np.array(Z[:, :, 0]), np.array(Z[:, :, 1]), np.array(Z[:, :, 2])])
    
    def _ungetZ2(self, Z2: list) -> list:
        """ @protected
        Transforms a (3*N, M) tensor to a (N, M, 3) tensor
        Reverses _getZ2
        """
        N, M = Z2.shape
        N //= 3
        
        Z = np.zeros((N, M, 3))
        Z[:, :, 0] = Z2[:N, :]
        Z[:, :, 1] = Z2[N:2*N, :]
        Z[:, :, 2] = Z2[2*N:, :]
        return Z
    
    def visualize(self) -> None:
        """ @public
        Visualize the results by plotting the following three images:
        - The original image, technically unaccessible
        - The in-painted image, technically the only one accessible
        - The corrected image
        """
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        fig.suptitle(self.name, fontsize=18)
        axs[0].title.set_text("Original Image")
        axs[0].imshow(self.image._get_image_original(), cmap='gray' if self.image.grayscale else 'viridis')
        axs[0].set_axis_off()
        axs[1].title.set_text(f"In-Painted Image ({self.image.ratio * 100} %)")
        axs[1].imshow(self.image._get_image_masked(), cmap='gray' if self.image.grayscale else 'viridis')
        axs[1].set_axis_off()
        axs[2].title.set_text(f"Corrected Image (rho={self.rho})")
        axs[2].imshow(self.sol.astype('float64'), cmap='gray' if self.image.grayscale else 'viridis')
        axs[2].set_axis_off()
        plt.show()