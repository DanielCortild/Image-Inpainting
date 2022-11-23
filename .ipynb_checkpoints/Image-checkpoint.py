from PIL import Image as ImagePIL
from PIL import ImageOps
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class Image:
    def __init__(self, 
                 image: str,
                 ratio: float, 
                 resize: tuple = False,
                 grayscale: bool = False) -> None:
        self.ratio = ratio
        self.resize = resize
        self.grayscale = grayscale
        
        self.image_original = ImagePIL.open(image)
        if self.resize: self.__resize(*resize)
        if self.grayscale: self.__grayscale()
        self.image_original = np.asarray(self.image_original, dtype=np.float64) / 255
        
        self.N, self.M = self.image_original.shape[:2]
        
        self.__create_mask()
        self.__mask_image()
        
    def __resize(self, width: int, height: int) -> None:
        """ @private
        Resize the image to a given size
        """
        self.image_original = self.image_original.resize((width, height))
        
    def __grayscale(self) -> None:
        """ @private
        Convert the image to grayscale (Single colour layer instead of 3)
        """
        self.image_original = ImageOps.grayscale(self.image_original)
        
    def __create_mask(self) -> None:
        """ @private
        Create the mask
        """
        self.mask_size = int(self.N * self.M * self.ratio)
        self.mask = (np.array([0] * self._get_nb_corrupt() + [1] * self._get_nb_correct()))
        np.random.shuffle(self.mask)
        self.mask = self.mask.reshape((self.N, self.M))
        
    def __mask_image(self) -> None:
        """ @private
        Create the masked image
        """
        self.image_masked = np.zeros_like(self.image_original)
        if len(self.image_original.shape) == 3:
            for i in range(3):
                self.image_masked[:, :, i] = np.multiply(self.image_original[:, :, i], self.mask)
        if len(self.image_original.shape) == 2:
            self.image_masked[:, :] = np.multiply(self.image_original[:, :], self.mask)
        
    def _get_image_original(self) -> list:
        """ @protected
        Get the original image
        """
        return self.image_original
    
    def _get_image_masked(self) -> list:
        """ @protected
        Get the masked image
        """
        return self.image_masked
    
    def _get_nb_corrupt(self) -> int:
        """ @protected
        Get the number of corrupt pixels
        """
        return self.mask_size
                     
    def _get_nb_correct(self) -> int:
        """ @protected
        Get the number of correct pixels
        """
        return self.N * self.M - self.mask_size
    
    def _get_mask(self) -> list:
        """ @protected
        Get the mask
        """
        return self.mask
    
    def visualize(self) -> None:
        """ @public
        Plotting the original and corrupted image
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 4), dpi=300)
        axs[0].title.set_text("Original Image")
        axs[0].imshow(self.image_original, cmap='gray' if self.grayscale else 'viridis')
        axs[0].set_axis_off()
        axs[1].title.set_text("In-Painted Image")
        axs[1].imshow(self.image_masked, cmap='gray' if self.grayscale else 'viridis')
        axs[1].set_axis_off()
        plt.show()