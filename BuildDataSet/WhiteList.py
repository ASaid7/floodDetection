#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:06:51 2020

@author: abdullahsaid
"""


from PIL import Image
import pandas as pd
import os
import numpy as np

'''
Clean data out based on distribution of file size, density of black pixels(zero values)
Amount of water and ignored pixels in the given tile 

'''

def mylistdir(directory):
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]
    
imgTiles = os.path.join(os.path.expanduser('~'),'floodDetection','Noaa Tiles')
maskTiles = os.path.join(os.path.expanduser('~'),'floodDetection','Mask Tiles')

files = mylistdir(imgTiles)
    
z=[os.stat(os.path.join(imgTiles,files[i])).st_size for i in range(len(files))]


df = pd.DataFrame(list(zip(files,z)), columns=['file','bytes'])
r = 50000
remove = df[df.bytes.astype(int)<=r].reset_index(drop=True)
keep = df[(df.bytes.astype(int)>r)].reset_index(drop=True)

z=[np.asarray(Image.open(os.path.join(imgTiles,keep.file[i]))).mean(
        ) for i in range(keep.shape[0])]
y = [np.sum(np.asarray(Image.open(os.path.join(imgTiles,keep.file[i])))==0
        ) for i in range(keep.shape[0])]

keep['meanpixel'] = z
keep['zerocount'] = y
keep['blackpercent'] = keep.zerocount/(1024*1024*3)

remove2=keep[(keep.blackpercent>.20)]
keep=keep[(keep.blackpercent>.20)]

bl = pd.concat([remove['file'],remove2['file']])

waterCnt = np.zeros(len(keep.file))
ignoreCnt = np.zeros(len(keep.file))

for i, j in enumerate(keep.file):
		img = np.asarray(Image.open(os.path.join(maskTiles,j[:-3]+'png')))
		waterCnt[i]=np.sum(img==1)
		ignoreCnt[i]=np.sum(img==2)


keep['waterCnt'] = waterCnt
keep['ignoreCnt'] = ignoreCnt

keep['waterPct'] = keep.waterCnt/(1024*1024)
keep['ignorePct'] = keep.ignoreCnt/(1024*1024)

remove3 = keep[(keep.waterPct<.001)&(keep.ignorePct>.400)]

pd.concat([bl['file'],remove3['file']])

keep = keep[(keep.waterPct>.001)&(keep.ignorePct<.400)]

keep.to_csv(os.path.join(os.path.dirname(imgTiles),'whiteList.csv'))




