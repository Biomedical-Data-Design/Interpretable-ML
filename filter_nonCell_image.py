#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:48:33 2023

@author: mikewang
"""
import cv2
import numpy as np
import os
from tqdm import tqdm
#%%
ImageFolderPath = 'extracted_cells/'
ImageTargetPath = 'filter_extracted_cells/'
#%%
def filter_non_cell_image(Path_source,Path_target):
    for image_name in tqdm(os.listdir(Path_source)):
        # read the image 
        image = cv2.imread(Path_source+image_name)
        image_x, image_y, channels = image.shape
        # linearlize the image 
        image_linear = image.reshape(image_x*image_y,channels)
        # calculating the color variance 
        image_r_var = np.var(image_linear[:,0])
        image_g_var = np.var(image_linear[:,1])
        image_b_var = np.var(image_linear[:,2])
        # average the variance
        image_var_avg = (image_r_var+image_g_var+image_b_var)/3
        # if the variance is large enough > 100 we consider it as a good cell image
        if image_var_avg > 200 and image_x*image_y > 1000: 
            cv2.imwrite(Path_target+'filtered_{}.png'.format(image_name), image)

#%%
filter_non_cell_image(ImageFolderPath, ImageTargetPath)
