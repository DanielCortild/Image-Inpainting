#!/usr/bin/env python
# encoding: utf-8
"""
Image.py - Implements an Image Class
~ Daniel Cortild, 26 November 2022
"""

# External Imports
from PIL import Image as ImagePIL     # type: ignore
import numpy as np
import matplotlib.pyplot as plt       # type: ignore
from typing import Tuple, Any, Union


class Image:
    """
    Provides a Class for a Masked Image, consisting of an image, a mask, and operations
    to modify the image and mask other images according to the same mask.
    Parameters:
        image                 Name of image, located in the same directory as code file
        ratio                 The ratio, between 0 and 1, of pixels to delete using the mask
        resize                If contains values, should contain a pair of integers, size
                              to which the image should be resize (Default: False)
    Public Methods:
        mask_image            Masks an image according to the created mask
        get_image_masked      Get the masked image
        visualize             Visualized the original, masked and corrected image
    Private Methods:
        resize                Resizes the image to a specific size
        create_mask           Creates the mask used to mask the image
    """
    
    def __init__(self, 
                 image_string: str,
                 image_dims: Tuple[int, int] = (256, 256)) -> None:
        self.image_dims = image_dims
        self.__load_image(image_string, image_dims)

    def __load_image(self, image_string: str, image_dims: Tuple[int, int]) -> None:
        """ @private
        Loads the image
        """
        self.image: Any = ImagePIL.open(image_string)
        self.image = self.image.resize(image_dims)
        self.image = np.asarray(self.image, dtype=np.float64) / 255

    def get_dimensions(self):
        """ @public
        Returns the dimensions of the image
        """
        return self.image_dims

    def get_image(self):
        """ @public
        Returns the image as an array
        """
        return self.image

    def show(self):
        """
        Prints the image to the console
        """
        fig, axs = plt.subplots(1, 1, figsize=(8, 8), dpi=600)
        axs.title.set_text("Original Image")
        axs.imshow(self.image, vmin=0, vmax=1)
        axs.set_axis_off()
        axs.set_facecolor("white")
        plt.show()

        
class MaskedImage (Image):
    """
    Creates a mask and methods to mask images, as well as getting the specific mask
    Parameters:
    Public Methods:
        mask_image                  Applies the mask to an image
        get_image_masked            Returns the masked image
        show                        Outputs the original and masked image
    Private Methods:
        create_mask                 Creates the mask to be applied
    """

    def __init__(self, image_string: str, image_size: Tuple[int, int] = (0, 0), erase_ratio: float = 0.5) -> None:
        super().__init__(image_string, image_size)
        self.__erase_ratio: float = erase_ratio
        self.create_mask()
        self.image_masked: np.ndarray = self.mask_image(self.get_image())

    def create_mask(self) -> np.ndarray:
        """ @private
        Create the mask
        """
        dimensions: List[int] = self.get_dimensions()
        total_pixels: int = np.prod(dimensions)
        mask_size: int = int(total_pixels * self.__erase_ratio)
        mask: np.ndarray = (np.array([0] * mask_size + [1] * (total_pixels - mask_size)))
        np.random.shuffle(mask)
        self.mask = mask.reshape(dimensions)

    def mask_image(self, image: np.ndarray) -> np.ndarray:
        """ @public
        Method encoding a linear operator selecting the pixels we know to be correct.
        Note this operator is self adjoint.
        """
        mask_3d: np.ndarray = np.repeat(self.mask[:, :, None], 3, axis=2)
        return np.multiply(image, mask_3d)

    def get_image_masked(self) -> np.ndarray:
        """ @public
        Get the masked image
        """
        return self.image_masked.copy()

    def get_erase_ratio(self) -> float:
        """ @public
        Returns the ratio of erased pixels
        """
        return self.__erase_ratio

    def show(self):
        """
        Prints the image to the console
        """
        fig, axs = plt.subplots(1, 2, dpi=600)
        axs[0].title.set_text("Original Image")
        axs[0].imshow(self.image, vmin=0, vmax=1)
        axs[0].set_axis_off()
        axs[1].title.set_text(f"Masked Image ({self.__erase_ratio * 100} %)")
        axs[1].imshow(self.image_masked, vmin=0, vmax=1)
        axs[1].set_axis_off()
        plt.show()

    def visualize(self,
                  solution1: Tuple[np.ndarray, str],
                  solution2: Union[Tuple[np.ndarray, str], None] = None,
                  title: str = "") -> None:
        """ @public
        Visualize the results by plotting the following three images:
        - The original image, technically unaccessible
        - The in-painted image, technically the only one accessible
        - The corrected image
        """
        fig, axs = plt.subplots(1, 4 if solution2 else 3, figsize=(12, 3.3 if solution2 else 4), dpi=600)
        fig.suptitle(title, fontsize=18)
        axs[0].title.set_text("Original Image")
        axs[0].imshow(self.image)
        axs[0].set_axis_off()
        axs[1].title.set_text(f"In-Painted Image ({self.__erase_ratio * 100} %)")
        axs[1].imshow(self.image_masked)
        axs[1].set_axis_off()
        axs[2].title.set_text(solution1[1])
        axs[2].imshow(solution1[0])
        axs[2].set_axis_off()
        if solution2:
            axs[3].title.set_text(solution2[1])
            axs[3].imshow(solution2[0])
            axs[3].set_axis_off()
        plt.show()

        
class DeletedImage (MaskedImage):
    """
    Creates a mask and methods to mask images, as well as getting the specific mask
    The image has a whole deleted section, as opposed to random deleted pixels
    Parameters:
    Public Methods:
        mask_image                  Applies the mask to an image
        get_image_masked            Returns the masked image
        show                        Outputs the original and masked image
    Private Methods:
        create_mask                 Creates the mask to be applied
    """

    def __init__(self, image_string: str, 
                 image_size: Tuple[int, int] = (0, 0)) -> None:
        super().__init__(image_string, image_size, 0)
        self.__erase_ratio: float = 0
        self.create_mask()
        self.image_masked: np.ndarray = self.mask_image(self.get_image())

    def add_block(self, x: float, y: float, z: float, w: float) -> None:
        """ @public
        Adds a block to the mask
        """
        dimensions: List[int] = self.get_dimensions()
        M, N = dimensions
        self.mask[int(x*M):int(y*M), int(z*N):int(w*N)] = np.zeros_like(self.mask[int(x*M):int(y*M), int(z*N):int(w*N)])
        self.image_masked: np.ndarray = self.mask_image(self.get_image())
        
    def create_mask(self) -> np.ndarray:
        """ @private
        Create the mask
        """
        dimensions: List[int] = self.get_dimensions()
        M, N = dimensions
        self.mask: np.ndarray = np.zeros((M, N)) + 1