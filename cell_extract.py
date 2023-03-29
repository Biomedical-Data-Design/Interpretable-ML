import cv2
import numpy as np
from tqdm import tqdm
import os

def extract_cell(filename, output_name):
    # Load image, grayscale, Otsu's threshold
    image = cv2.imread(filename)
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
    # filter out all over lapping bounding boxs
    # first store the bbox coordinate
    for i, c in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c) # bounding box parameters
        bbox_array[:,i] = np.array([x,y,x+w,y+h])
    # second filter out all overlapping cases
    for i in range(cnts_length):
        if overlapped_check_array[i] != 0:
            x_tl,y_tl,x_br,y_br = bbox_array[:,i] # current bbox
            w = x_br-x_tl
            h = y_br - y_tl
            if w/h > 1.5 or h/w > 1.5:
                overlapped_check_array[i] = 0
            # loop through all bbox
            for j in range(cnts_length):
                if overlapped_check_array[j] != 0 and overlapped_check_array[i] != 0 and i != j:
                    x_tl_loop,y_tl_loop,x_br_loop,y_br_loop = bbox_array[:,j] # looped troughed bbox
                    # if not not overlapping
                    if not ((x_tl > x_br_loop or x_tl_loop > x_br) or (y_br > y_tl_loop or y_br_loop > y_tl)):
                        overlapped_check_array[i] = 0
                        overlapped_check_array[j] = 0

    bbox_array_non_overlapping = bbox_array[:,overlapped_check_array==1]
    non_overlapping_length = bbox_array_non_overlapping.shape[1]
    # extract only non-overlapping bbox
    for i in range(non_overlapping_length):
        x_tl,y_tl,x_br,y_br = bbox_array_non_overlapping[:,i]
        cv2.rectangle(image, (int(x_tl), int(y_tl)), (int(x_br), int(y_br)), (36,255,12), 2)
        ROI = original[int(y_tl):int(y_br), int(x_tl):int(x_br)]
        cv2.imwrite('{}_ROI_{}.png'.format(output_name, ROI_number), ROI)
        ROI_number += 1
    return


if __name__ == "__main__":
    folder = "/Users/yukuai/Documents/EN.580.637/Project_Shapley/code/my_code/malaria/images"
    for filename in os.listdir(folder):
        image_filename = os.path.join(folder, filename)
        print(image_filename)
        extract_cell(image_filename, "./extracted_cells/" + filename)

#%%
