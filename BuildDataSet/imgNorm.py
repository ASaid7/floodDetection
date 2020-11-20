#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:22:44 2020

@author: abdullahsaid
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

'''
Find mean and std of training image/hand to normalize for model
'''

imgPath = os.path.join(os.path.expanduser('~'),'floodDetection','Noaa Tiles')
maskPath = os.path.join(os.path.expanduser('~'),'floodDetection','Mask Tiles')
handPath = os.path.join(os.path.expanduser('~'),'floodDetection','HAND Tiles')


df = pd.read_csv(os.path.join(os.path.dirname(imgPath),'train.csv'))
n = len(df)
means = np.zeros((n,4))
stds = np.zeros((n,4))
for j,i in enumerate(df.file):
    img = np.asarray(Image.open(os.path.join(imgPath,i+".jpg")))/255
    hnd = np.asarray(Image.open(os.path.join(handPath,i+'.tif')))
    msk = np.asarray(Image.open(os.path.join(maskPath,i+'.png')))
    locImg = np.argwhere(msk!=2)
    means[j,:3] = img[locImg[:,0],locImg[:,1]].mean(axis=0)
    means[j,3] = hnd[locImg[:,0],locImg[:,1]].mean()
    stds[j, :3] = img[locImg[:,0],locImg[:,1]].std(axis=0)
    stds[j,3] = hnd[locImg[:,0],locImg[:,1]].std()


with open(os.path.join(os.path.dirname(imgPath),'Norm.npy', 'wb')) as f:
    np.save(f, means.mean(axis=0))
    np.save(f, stds.mean(axis=0))
    
