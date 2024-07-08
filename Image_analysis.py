#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:18:23 2023

@author: lucafaccenda
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io,color,filters,measure
from skimage.morphology import reconstruction
import scipy.ndimage as nd
from skimage.segmentation import clear_border






def process(image):
    #get grayscale
    image_gray = color.rgb2gray(image)
    selected_channel = image[:,:,2]
    
    #find threshold
    thmin = filters.threshold_minimum(selected_channel)
    thmean = filters.threshold_mean(selected_channel)
    global_threshold = filters.threshold_isodata(selected_channel)
    
    
    #Create Mask to separate Background and forground
    rawmask = selected_channel<global_threshold
    
    
    #Remove objects touching the border
    no_back_or_border =  clear_border(rawmask)
    
    #filling in spaces in mask
    source = np.copy(no_back_or_border)
    source[1:-1, 1:-1] = no_back_or_border.max()
    mask = no_back_or_border

    filled = reconstruction(source, mask, method='erosion')
    
    
    labels = measure.label(filled)

    #APPLY MASK
    l = []
    for i in range(3):
        q = (image[:,:,i]*filled)
        l.append(q)
     
    
    #Use threshold to remove background
    no_background_img = np.dstack([l[0],l[1],l[2]])
    
    

    
    
    
    
    
    return image_gray,thmean,global_threshold, rawmask , no_background_img,filled,labels,no_back_or_border



image = io.imread('20230113_102106.jpg') 




image_gray, thmean,global_threshold, rawmask , no_background_img,filled,labels,no_back_or_border = process(image)

print(labels.max())

b = no_back_or_border.astype('uint8')






