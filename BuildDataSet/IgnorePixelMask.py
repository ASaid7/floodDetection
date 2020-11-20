#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:40:03 2020

@author: abdullahsaid
"""

import os
import numpy as np
import cv2
from PIL import Image

'''
ignores any padded areas in the tiles of NOAA and awkward crops from NOAA images
'''

imgTiles = os.path.join(os.path.expanduser('~'),'floodDetection','Noaa Tiles')
maskTiles = os.path.join(os.path.expanduser('~'),'floodDetection','Mask Tiles')


imgList = [i[:-4] for i in os.listdir(imgTiles)]
labelList = [i[:-4] for i in os.listdir(maskTiles)]

u = np.zeros((44*1024,2))
d = np.zeros((44*1024,2))
l = np.zeros((44*1024,2))
r = np.zeros((44*1024,2))
c=0
for i in range(44):
    for j in range(1024):
        u[c] = [i,j]
        d[c] = [i+980,j]
        l[c] = [j,i]
        r[c] = [j,i+980]
        c+=1
u = u.astype('int')
d = d.astype('int')
l = l.astype('int')
r = r.astype('int')

for i in labelList:
    img = np.asarray(Image.open(os.path.join(imgTiles,i+'.jpg')))
    mask = np.asarray(Image.open(os.path.join(maskTiles,i+'.png')))
    cpMask = mask.copy()
    cpMask = (cpMask==255)*1.
    locImg = np.argwhere(img.mean(axis=0)==2)
    cpMask[locImg[:,0], locImg[:,1]]=2.
    if i[-3:-2] == '9':
        cpMask[u[:,0],u[:,1]] = 2.
        cpMask[d[:,0],d[:,1]] = 2.
    if i[-1:] == '9':
        cpMask[l[:,0],l[:,1]] = 2.
        cpMask[r[:,0],r[:,1]] = 2.
    cv2.imwrite(os.path.join(maskTiles,i+'.png'),cpMask)