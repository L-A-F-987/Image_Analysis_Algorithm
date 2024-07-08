#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:02:51 2023

@author: 2572705
"""

#Importing Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import io,filters
import scipy.ndimage as nd
from skimage.segmentation import clear_border
from scipy.spatial import distance as dist

# cv2 is imported to help with image processing and manipulation
import cv2 as cv

#Imutils is used to extract contours from the cv2 contours making it easier to separate objects
import imutils
from imutils import perspective

def process(img):
    
    #Reading in image
    im = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

     #Converting image from bgr to rgb
    img1 = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    
    #Finding threshold for mask
    val = filters.threshold_otsu(im)
    
    #creating binary mask for background
    drops = nd.binary_fill_holes(im < val)
    
    #Remove objects touching the border
    #Opened first to prevent accidental removal where nonskittle background
    #noise touches side causing skittle to be removed
    
    
    #Setup Kernal as a 30,30 matrix of ones for errosion
    pre_kernel = np.ones((30,30),np.uint8)
    
    #Applying Open to image
    temp_drops = cv.morphologyEx(drops.astype('uint8'), cv.MORPH_OPEN, pre_kernel)

    #Crearing a mask to remove objects in contact with border and background
    no_back_or_border =  clear_border(temp_drops)
    
    
    #Applying mask to each colour
    non_stacked_img = []
    for i in range(3):
        temp = img1[:,:,i]*no_back_or_border
        non_stacked_img.append(temp)
    
    
    #restacking each color to reform image
    img1_masked = np.dstack((non_stacked_img[0],non_stacked_img[1],non_stacked_img[2]))
    
    #grayscaling masked image
    masked_gray = cv.cvtColor(img1_masked, cv.COLOR_RGB2GRAY)

    #Bluring grayscale to make contours easier to find by removing small details
    gray_blur = cv.medianBlur(masked_gray,25)

    #Finds the outline of an Object using cotours
    cnt = cv.findContours(gray_blur,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnt)
    

        
    #Creating a function to find the midpoint of the rectangle that will be drawn around objects
    def midpoint(ptA, ptB):
    	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    #Creating empty arrays to be filled
    boxes = []
    new_cnts = []


    #Creating Array of ones the same size as the image to be used later
    mask = np.ones(im.shape[:2], dtype="uint8")*255


    #For loop to find centroids and object shape
    for c in cnts: 
        #Creates an Array called box that is a rectangle around an object at its most extreme dimentions
        box = cv.minAreaRect(c)
        box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
        #Converts the box values to integers as they should be for pixel values 
        box = np.array(box, dtype="int")
        #orders the points in the box varriable
        box = perspective.order_points(box)
        
        # creates varriables for each corner of the box 
        (tl, tr, br, bl) = box
        #Finds the midpoint between each corner using the midpoint fuction created
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        
        #Calculates the width and length of the object by measuring distance between midpoints
        Width = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        Length = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        #Assumes that all desired objects are round and therefore 
        #the object should be as wide as it is tall therefore if it's
        #dimention ratio is >2.5:1 it will stored in a separate array (new_cnts)
        
        if Length<Width*2.5 and Width<Length*2.5:
            boxes.append(box)
            
        
        else:
            #Creating a mask to remove the long objects
            cv.drawContours(mask, [c], -1, 0, -1)
            new_cnts.append(c)
         
    #Converting mask to uint8 
    mask = mask.astype('uint8')
        
          
    #applying mask of long objects
    Long_objects_removed = cv.bitwise_and(img1_masked, img1_masked, mask = mask)
        
    #Applying Inverted Mask to Check What Was Removed
    inverted_mask = cv.bitwise_not(mask)
    
    
    Removed_Object_Img = cv.bitwise_and(img1_masked, img1_masked, mask = inverted_mask)


    #Converting removed object image to hsv
    Removed_Object_hsv = cv.cvtColor(Removed_Object_Img, cv.COLOR_RGB2HSV)
    
        
    #Color Thresholding Removed Image 
    
    brown_long = cv.inRange(Removed_Object_hsv,(95,0,0),(250,190,95))

    green_long = cv.inRange(Removed_Object_hsv,(35,25,25),(80,255,255))
    
        
    #Adding Back any skittles that may have been removed by accident
    final_image_mask = green_long+brown_long
    
    
    #Creating kernel for closing 'final image'
    kernel = np.ones((50,50),np.uint8)
    
    #Closing final_image_mask to fill in empty areas within objects eg. bright spots
    final_image_mask = cv.morphologyEx(final_image_mask, cv.MORPH_CLOSE, kernel)
    
    
    #isolating objects to be added back to image that were removed if touching long objects
    to_add_back = cv.bitwise_and(img1_masked, img1_masked, mask = final_image_mask)
    
    #Summing the 2 images to get final image
    final_image = to_add_back + Long_objects_removed
    final_image_hsv=  cv.cvtColor(final_image, cv.COLOR_RGB2HSV)
    
    

    #Creating mask to separate green and brown skittls in the final image
    brown_mask_for_count = cv.inRange(final_image_hsv,(100,0,0),(250,250,150))
    
    green_mask_for_count = cv.inRange(final_image_hsv,(35,25,25),(80,255,255))
    
    #Applying masks
    brown_iso =  cv.bitwise_and(final_image, final_image, mask = brown_mask_for_count)
    green_iso =  cv.bitwise_and(final_image, final_image, mask = green_mask_for_count)
    
    
    #Blurring image to prevent small bright objects being counted
    gray_green_iso = cv.medianBlur(cv.cvtColor(green_iso, cv.COLOR_RGB2GRAY),25)
    gray_brown_iso = cv.medianBlur(cv.cvtColor(brown_iso, cv.COLOR_RGB2GRAY),25)

    #Removing Small Objects
    gray_brown_iso = cv.morphologyEx(gray_brown_iso, cv.MORPH_OPEN, kernel)
    gray_green_iso = cv.morphologyEx(gray_green_iso, cv.MORPH_OPEN, kernel)
    
    #Eroding to remove touching objects 
    erode_kernel = np.ones((25,25),np.uint8)
    

    gray_brown_iso = cv.erode(gray_brown_iso, erode_kernel, iterations=1)
    gray_green_iso = cv.erode(gray_green_iso, erode_kernel, iterations=1)

   
    
    #Finding Centroids
    cnts_green = cv.findContours(gray_green_iso,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    cnts_green = imutils.grab_contours(cnts_green)
    
    cnts_brown = cv.findContours(gray_brown_iso,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    cnts_brown = imutils.grab_contours(cnts_brown)
    
    
    #Counting number of centroids to be returned from function
    number_1 = len(cnts_green)
    
    number_2 = len(cnts_brown)
    


    
    coordinates_1 = []
    coordinates_2 = []
    
    #filling list with centroid values when number of objects is > 0
    if number_1 >0:  
        for i in cnts_green:
            M = cv.moments(i)
            iX = int(M["m10"] / M["m00"])
            iY = int(M["m01"] / M["m00"])
            cent = [iX,iY]
            coordinates_1.append(cent)
         
    if number_2 >0:
        for i in cnts_brown:
            M = cv.moments(i)
            iX = int(M["m10"] / M["m00"])
            iY = int(M["m01"] / M["m00"])
            cent = [iX,iY]
            coordinates_2.append(cent)
        
    #Returning required varriable
    return number_1,number_2,coordinates_1,coordinates_2


def highlight(image,coordinates_1,coordinates_2,ax=None):
    #do not change this function
    if ax is None:
        fig,ax = plt.subplots()
    ax.imshow(image)
    for x,y in coordinates_1:
        ax.plot(x,y,'ro')
    for x,y in coordinates_2:
        ax.plot(x,y,'wo')



card = []
#do not remove the line below and put all your runtime code down there so that this file can be imported without executing anything
if __name__ == "__main__":
    #read an image
    with open('image_directory.csv','r') as M:
        Normal = M.readlines()
        
    for i in range(len(Normal)):
        N = Normal[i]
        N=N.replace('"','')
        N=N.replace('\n','')
        card.append(N)

        
    for i in range(len(card)-12,len(card)):
        image = io.imread(card[i])
        #get the numbers from the function
        n1,n2,coo1,coo2 = process(image)
        #plot it
        highlight(image,coo1,coo2)
        plt.title(card[i])
        plt.show()
        print(len(Normal)-i)












