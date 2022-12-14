#!/usr/bin/env python
# encoding: utf-8
"""
Image.py - Implements an Image Class with Masking Options
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
        ratio                 Ratio, between 0 and 1, of pixels to delete using the mask
        resize                If contains values, should contain a pair of integers, size
                              to which the image should be resize (Default: False)
    Public Methods:
        mask_image            Masks an image according to the created mask
        get_image_masked      Get the masked image
        getZ1                 Get the transformed image into a specific form
        ungetZ1               Undoes getZ1
        getZ2                 Get the transformed image into a specific form
        ungetZ2               Undoes getZ2
        visualize             Visualized the original, masked and corrected image
    Private Methods:
        resize                Resizes the image to a specific size
        create_mask           Creates the mask used to mask the image
    """
    
    def __init__(self, 
                 image: str,
                 ratio: float, 
                 resize: Tuple[int, int] = (0, 0)) -> None:
        
        # Transform Image according to the parameters
        self.image_pre: Any = ImagePIL.open(image)
        if resize != (0, 0): self.__resize(*resize)
        self.image_original: np.ndarray = np.asarray(self.image_pre, dtype=np.float64) / 255
        
        # Mask image according to the ratio parameter
        self.ratio: float = ratio
        self.mask: np.ndarray = self.__create_mask(self.image_original.shape[0], self.image_original.shape[1], ratio)
        self.image_masked: np.ndarray = self.mask_image(self.image_original)
       
    
    def __resize(self, width: int, height: int) -> None:
        """ @private
        Resize the image to a given size
        """
        self.image_pre = self.image_pre.resize((width, height))
        
        
    @staticmethod
    def __create_mask(M: int, N: int, ratio: float) -> np.ndarray:
        """ @private
        Create the mask
        """
        mask_size: int = int(N * M * ratio)
        mask: np.ndarray = (np.array([0] * mask_size + [1] * (M * N - mask_size)))
        np.random.shuffle(mask)
        return mask.reshape((N, M))
       
        
    def mask_image(self, img: np.ndarray) -> np.ndarray:
        """ @public
        Method encoding a linear operator selecting the pixels we know to be correct.
        This operator is self adjoint, and could be coded to return a vector instead of a matrix, 
        but that would not change the maths, and the opposite simplifies the coding.
        """
        img_masked: np.ndarray = np.zeros_like(img)
        for i in range(3):
            img_masked[:, :, i] = np.multiply(img[:, :, i], self.mask)
        return img_masked
    
    
    def get_image_masked(self) -> np.ndarray:
        """ @public
        Get the masked image
        """
        return self.image_masked
    
    
    @staticmethod
    def getZ1(Z: np.ndarray) -> np.ndarray:
        """ @public
        Transforms a (N, M, 3) tensor to a (N, 3*M) tensor
        Reverses ungetZ1
        """
        return np.hstack([np.array(Z[:, :, 0]), np.array(Z[:, :, 1]), np.array(Z[:, :, 2])])

    
    @staticmethod
    def ungetZ1(Z1: np.ndarray) -> np.ndarray:
        """ @public
        Transforms a (N, 3*M) tensor to a (N, M, 3) tensor
        Reverses getZ1
        """
        N: int; M:int
        N, M = Z1.shape
        M //= 3
        
        Z: np.ndarray = np.zeros((N, M, 3))
        Z[:, :, 0] = Z1[:, :M]
        Z[:, :, 1] = Z1[:, M:2*M]
        Z[:, :, 2] = Z1[:, 2*M:]
        return Z
    
    
    @staticmethod
    def getZ2(Z: np.ndarray) -> np.ndarray:
        """ @public
        Transforms a (N, M, 3) tensor to a (3*N, M) tensor
        Reverses ungetZ2
        """
        return np.vstack([np.array(Z[:, :, 0]), np.array(Z[:, :, 1]), np.array(Z[:, :, 2])])
    
    
    @staticmethod
    def ungetZ2(Z2: np.ndarray) -> np.ndarray:
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
        axs[0].imshow(self.image_original)
        axs[0].set_axis_off()
        axs[1].title.set_text(f"In-Painted Image ({self.ratio * 100} %)")
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