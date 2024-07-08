#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:56:19 2023

@author: lucafaccenda
"""

card = []
#do not remove the line below and put all your runtime code down there so that this file can be imported without executing anything
if name == "main":
    #read an image
    with open('image_directory.csv','r') as M:
        Normal = M.readlines()

    for i in range(len(Normal)):
        N = Normal[i]
        N=N.replace('"','')
        N=N.replace('\n','')
        card.append(N)


    for i in range(len(card)):
        image = io.imread(card[i])
        #get the numbers from the function
        n1,n2,coo1,coo2 = process(image)
        #plot it
        highlight(image,coo1,coo2)
        plt.title(card[i])
        plt.show()
        print(len(Normal)-i) 
