# -*- coding: utf-8 -*-

"""
Skeleton for excersize on edge operators
Student: Stian Teien
"""

from skimage.viewer import ImageViewer
from skimage import io
from skimage import util
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import math



def edge_operator(image, operator):
    """Returns the reusult from one of the edge operators, prewitt, sobel,
    canny eller laplace
    
    Parameters:
    -----------
    image : np.ndarray
        Image to detect blobs in. If this image is a colour image then 
        the last dimension will be the colour value (as RGB values).
    operator : numeric
    1 = sobel filter
    2 = prewitt filter
    3 = canny filter
    4 = laplace filter

    Returns:
    --------
    filtered : np.ndarray(np.uint)
    result image from the edge operator
    """
    
    pic = util.img_as_ubyte(image) 
    
    # Sobel filter
    if(operator == 1):
        vertical_mask = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        horsintal_mask = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        
        
        v_pic = np.zeros((pic.shape[0], pic.shape[1]))
        h_pic = np.zeros((pic.shape[0], pic.shape[1]))
        egde_pic = np.zeros((pic.shape[0], pic.shape[1]))
        
        # Iterate image
        for i in range(pic.shape[0]-1):
            for j in range(pic.shape[1]-1):
                temp_v_pic = np.zeros((3,3))
                temp_h_pic = np.zeros((3,3))
                for o in range(-1,2,1):
                    for k in range(-1,2,1):
                        temp_v_pic[o+1,k+1] = (pic[i+o,j+k] * vertical_mask[o+1][k+1])/9
                        temp_h_pic[o+1,k+1] = (pic[i+o,j+k] * horsintal_mask[o+1][k+1])/9
                        
                
                v_pic[i,j] = pic[i,j] * sum(sum(temp_v_pic))
                h_pic[i,j] = pic[i,j] * sum(sum(temp_h_pic))
                egde_pic[i,j] = math.sqrt(v_pic[i,j]**2 + h_pic[i,j]**2)
        
        
                
        
        
        
    return egde_pic, v_pic, h_pic # Filtered


def sharpen(image, sharpmask):
    """Performs an image sharpening using Laplace filter or unsharpen mask (USM)
    1 = Laplace
    2 = USM
    
    Returns: sharpened image
    """
    pic = rgb2gray(image)
    
    # Laplace
    if(sharpmask == 1):
        vertical_mask = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        horsintal_mask = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        
        v_pic = np.zeros((pic.shape[0], pic.shape[1]))
        h_pic = np.zeros((pic.shape[0], pic.shape[1]))
        sharp_pic = np.zeros((pic.shape[0], pic.shape[1]))
        
        # Iterate image
        for i in range(pic.shape[0]-1):
            for j in range(pic.shape[1]-1):
                temp_v_pic = np.zeros((3,3))
                temp_h_pic = np.zeros((3,3))
                for o in range(-1,2,1):
                    for k in range(-1,2,1):
                        temp_v_pic[o+1,k+1] = (pic[i+o,j+k] * vertical_mask[o+1][k+1])/9
                        temp_h_pic[o+1,k+1] = (pic[i+o,j+k] * horsintal_mask[o+1][k+1])/9
                        
                
                v_pic[i,j] = pic[i,j] * sum(sum(temp_v_pic))
                h_pic[i,j] = pic[i,j] * sum(sum(temp_h_pic))
                sharp_pic[i,j] = math.sqrt(v_pic[i,j]**2 + h_pic[i,j]**2)
        
        sharp_pic = pic - sharp_pic
        
    return sharp_pic, v_pic, h_pic



picture = io.imread('IR_Heraklion.tif')
picture_unsharp = io.imread('hot_image.png')
egde_pic = edge_operator(picture, 1)
sharp_pic = sharpen(picture_unsharp, 1)

plt.imsave("egde.png",egde_pic[0])
plt.imsave("sharp.png",sharp_pic[0])

#plt.imshow(sharp_pic[0])

