#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:21:33 2020

@author: abdullahsaid
"""

import cv2
import math
import os
import numpy as np
from osgeo import gdal
import pandas as pd


def mylistdir(directory):
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

raw = os.path.join(os.path.expanduser('~'),'floodDetection','Raw Noaa Images')
mask = os.path.join(os.path.expanduser('~'),'floodDetection','Mask')
hand = os.path.join(os.path.expanduser('~'),'floodDetection','harvey_hand_float.tif')
csv = os.path.join(os.path.expanduser('~'),'floodDetection','NoaaCoords.csv')

imgTiles = os.path.join(os.path.expanduser('~'),'floodDetection','Noaa Tiles')
maskTiles = os.path.join(os.path.expanduser('~'),'floodDetection','Mask Tiles')
handTiles = os.path.join(os.path.expanduser('~'),'floodDetection','HAND Tiles')

folder(imgTiles)
folder(maskTiles)
folder(handTiles)
    
size = (1024, 1024)
offset = (935, 935)

def tiles(path, tileSize, offset, extension, save, csv=None):
    if csv is not None:
        files = pd.read_csv(csv)
    else:
        files = mylistdir(path)
    for i in files:
        if csv is not None:
                n,s,e,w = i[4], i[2], i[3], i[1]
                img = gdal.Translate(os.path.join(os.path.dirname(path),'_.tif'), path, projWin = [w,n,e,s], width=9351, height=9351, resampleAlg='cubic',
                         strict=True)
                img = img.ReadAsArray()
                img[img < 0] = 0
        else:
            img = cv2.imread(os.path.join(path,i))
        img_shape = img.shape
        if img_shape[2]==3:
            pad0=((44,44),(0,0),(0,0))
            pad1=((0,0),(44,44),(0,0))
        else:
            pad0=((44,44),(0,0))
            pad1=((0,0),(44,44))
        for j in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
            for k in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
                cropped_img = img[offset[1]*j:min(offset[1]*j+tileSize[1], img_shape[0]), 
                                  offset[0]*k:min(offset[0]*k+tileSize[0], img_shape[1])]
                if cropped_img.shape[0] == 936:
                    cropped_img = np.pad(cropped_img,pad0)
                if cropped_img.shape[1] == 936:
                    cropped_img = np.pad(cropped_img,pad1)
                #Debugging the tiles
                print(i,cropped_img.shape)
                if cropped_img.shape==(1024,1024,3):
                    if extension == '.tif':
                        cropped_img=cropped_img.astype('float32')
                    cv2.imwrite(os.path.join(save,i[:-4]+'_'+ str(j) + "_" + str(k) + extension), cropped_img)
                    
                    

tiles(raw,size, offset, '.jpg', imgTiles)
tiles(mask, size, offset, '.png', maskTiles)
tiles(hand, size, offset, '.tif', handTiles, csv)