#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:26:12 2023

@author: mikewang
"""
import cv2
import numpy as np
from tqdm import tqdm
import os

#%% find the bbox from a whole image
# Load image, grayscale, Otsu's threshold
image = cv2.imread('1f8f08ea-b5b3-4f68-94d4-3cc071b7dce8.png')
Path_bagOfCells = 'filter_extracted_cells/'
Path_patchedimage = 'patched_examples/'
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = 255 - gray
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Morph open to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours, obtain bounding box, extract and save ROI
ROI_number = 0
cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

cnts_length = len(cnts)
bbox_array = np.zeros([4,cnts_length])
overlapped_check_array = np.ones([cnts_length,], dtype = 'int32')


#%% check the relative size and the color of the cells in the image
# first store the bbox coordinate
for i, c in enumerate(cnts):
    x,y,w,h = cv2.boundingRect(c) # bounding box parameters
    bbox_array[:,i] = np.array([x,y,w,h])

# calculate the relateive size of the bbox 
bbox_h_var = np.var(bbox_array[3,:])
bbox_w_var = np.var(bbox_array[2,:])
bbox_h_mean = np.mean(bbox_array[3,:])
bbox_w_mean = np.mean(bbox_array[2,:])


# rgb mean and var array 
rgb_mean_var_array = np.zeros([6,len(cnts)])
for i, c in enumerate(cnts):
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    ROI = original[y:y+h, x:x+w]
    ROI_x, ROI_y, channels = ROI.shape
    ROI_linear = ROI.reshape(ROI_x*ROI_y, channels)
    rgb_mean_var_array[0,i], rgb_mean_var_array[3,i] = np.mean(ROI_linear[:,0]), np.var(ROI_linear[:,0])
    rgb_mean_var_array[1,i], rgb_mean_var_array[4,i] = np.mean(ROI_linear[:,1]), np.var(ROI_linear[:,1])
    rgb_mean_var_array[2,i], rgb_mean_var_array[5,i] = np.mean(ROI_linear[:,2]), np.var(ROI_linear[:,2])
rgb_mean_var = np.mean(rgb_mean_var_array,axis=1)

#%% matching for relatively similiar color and size cell in the bag 
cell_image_names = []

for name in os.listdir(Path_bagOfCells):
    if ".png" in name: 
        cell_image_names.append(name)


num_cell = len(cell_image_names)
cell_index_array = np.array(range(num_cell))
np.random.shuffle(cell_index_array)

# find the matching cell image
for i in range(num_cell):
    cell_index = cell_index_array[i]
    cell_image_name = cell_image_names[cell_index]
    cell_image = cv2.imread(Path_bagOfCells+cell_image_name)
    cell_image_x, cell_image_y, cell_image_channel = cell_image.shape
    cell_image_linear = cell_image.reshape([cell_image_x*cell_image_y, cell_image_channel])
    cell_image_r_mean, cell_image_r_var = np.mean(cell_image_linear[:,0]), np.var(cell_image_linear[:,0])
    cell_image_g_mean, cell_image_g_var = np.mean(cell_image_linear[:,1]), np.var(cell_image_linear[:,1])
    cell_image_b_mean, cell_image_b_var = np.mean(cell_image_linear[:,2]), np.var(cell_image_linear[:,2])
    cell_image_color_mean_var_array = np.array([cell_image_r_mean, cell_image_g_mean, cell_image_b_mean, cell_image_r_var, cell_image_g_var, cell_image_b_var])
    error = np.linalg.norm(rgb_mean_var-cell_image_color_mean_var_array)
    if error < 1:
        break

#%% patching 
# example find one of the bbox and replace it with a patch
c_example = cnts[20]
x,y,w,h = cv2.boundingRect(c_example) # bounding box location in the picture 
cell_image = cv2.resize(cell_image, (w,h))
patched_image = original.copy()
patched_image[y:y+h, x:x+w] = cell_image

cv2.imwrite(Path_patchedimage+'original_{}.png'.format(cell_image_name), original)
cv2.imwrite(Path_patchedimage+'patched_{}.png'.format(cell_image_name), patched_image)








