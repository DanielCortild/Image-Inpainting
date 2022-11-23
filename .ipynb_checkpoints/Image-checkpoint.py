from PIL import Image as ImagePIL
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class Image:
    """
    Provides a Class for a Masked Image, consisting of an image, a mask, and operations
    to modify the image and mask other images according to the same mask.
    Parameters:
        image                 Name of image, located in the same directory as code file
        ratio                 Ratio, between 0 and 1, of pixels to delete using the mask
        resize                If contains values, should contain a pair of integers, size
                              to which the image should be resize (Default: False)
    Public Methods:
        visualize             Visualized the original, masked and corrected image
        mask_image            Masks an image according to the created mask
        getZ1                 Get the transformed image into a specific form
        ungetZ1               Undoes getZ1
        getZ2                 Get the transformed image into a specific form
        ungetZ2               Undoes getZ2
        get_image_masked      Get the masked image
    Protected Methods:
    Private Methods:
        resize                Resizes the image to a specific size
        create_mask           Creates the mask used to mask the image
    """
    
    def __init__(self, 
                 image: str,
                 ratio: float, 
                 resize: tuple = False) -> None:
        
        # Transform Image according to the parameters
        self.image_original = ImagePIL.open(image)
        if resize: self.__resize(*resize)
        self.image_original = np.asarray(self.image_original, dtype=np.float64) / 255
        
        # Mask image according to the ratio parameter
        self.ratio = ratio
        self.mask = self.__create_mask(*self.image_original.shape[:2], ratio)
        self.image_masked = self.mask_image(self.image_original)
        
    def __resize(self, width: int, height: int) -> None:
        """ @private
        Resize the image to a given size
        """
        self.image_original = self.image_original.resize((width, height))
        
    def __create_mask(self, M: int, N: int, ratio: float) -> None:
        """ @private
        Create the mask
        """
        mask_size = int(N * M * ratio)
        mask = (np.array([0] * mask_size + [1] * (M * N - mask_size)))
        np.random.shuffle(mask)
        mask = mask.reshape((N, M))
        return mask
        
    def mask_image(self, img: list) -> list:
        """ @public
        Method encoding a linear operator selecting the pixels we know to be correct.
        This operator is self adjoint, and could be coded to return a vector instead of a matrix, 
        but that would not change the maths, and the opposite simplifies the coding.
        """
        img_masked = np.zeros_like(img)
        for i in range(3):
            img_masked[:, :, i] = np.multiply(img[:, :, i], self.mask)
        return img_masked
    
    def get_image_masked(self) -> list:
        """ @public
        Get the masked image
        """
        return self.image_masked
    
    def getZ1(self, Z: list) -> list:
        """ @public
        Transforms a (N, M, 3) tensor to a (N, 3*M) tensor
        Reverses ungetZ1
        """
        return np.hstack([np.array(Z[:, :, 0]), np.array(Z[:, :, 1]), np.array(Z[:, :, 2])])

    def ungetZ1(self, Z1: list) -> list:
        """ @public
        Transforms a (N, 3*M) tensor to a (N, M, 3) tensor
        Reverses getZ1
        """
        N, M = Z1.shape
        M //= 3
        
        Z = np.zeros((N, M, 3))
        Z[:, :, 0] = Z1[:, :M]
        Z[:, :, 1] = Z1[:, M:2*M]
        Z[:, :, 2] = Z1[:, 2*M:]
        return Z
    
    def getZ2(self, Z: list) -> list:
        """ @public
        Transforms a (N, M, 3) tensor to a (3*N, M) tensor
        Reverses ungetZ2
        """
        return np.vstack([np.array(Z[:, :, 0]), np.array(Z[:, :, 1]), np.array(Z[:, :, 2])])
    
    def ungetZ2(self, Z2: list) -> list:
        """ @public
        Transforms a (3*N, M) tensor to a (N, M, 3) tensor
        Reverses getZ2
        """
        N, M = Z2.shape
        N //= 3
        
        Z = np.zeros((N, M, 3))
        Z[:, :, 0] = Z2[:N, :]
        Z[:, :, 1] = Z2[N:2*N, :]
        Z[:, :, 2] = Z2[2*N:, :]
        return Z
    
    def visualize(self, solution: list, title: str = "") -> None:
        """ @public
        Visualize the results by plotting the following three images:
        - The original image, technically unaccessible
        - The in-painted image, technically the only one accessible
        - The corrected image
        """
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        fig.suptitle(title, fontsize=18)
        axs[0].title.set_text("Original Image")
        axs[0].imshow(self.image_original)
        axs[0].set_axis_off()
        axs[1].title.set_text(f"In-Painted Image ({self.ratio * 100} %)")
        axs[1].imshow(self.image_masked)
        axs[1].set_axis_off()
        axs[2].title.set_text(f"Corrected Image")
        axs[2].imshow(solution)
        axs[2].set_axis_off()
        plt.show()