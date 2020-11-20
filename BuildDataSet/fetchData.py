#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:51:55 2020

@author: abdullahsaid
"""

import os
import requests
import tarfile
from tqdm import tqdm
from osgeo import gdal
import pandas as pd

def folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def fetcher(urls, path):
    folder(path)
    print('Downloading {} files'.format(len(urls)))
    for url in tqdm(urls):
        image_name = url.split("/")[-1]
        image_path = os.path.join(path, image_name)

        if not os.path.isfile(image_path):
            response=requests.get(url,stream=True)

            with open(image_path,'wb') as outfile:
                outfile.write(response.content)
                
def extractor(path, extension='.tar'):
    folder(os.path.join(noaaPath))
    print('Files Extraction')
    for file in tqdm(os.listdir(path)):
        if file.endswith(extension):
            tarfile.open(os.path.join(path,file)).extractall(noaaPath)
            os.remove(os.path.join(path,file))

            
def coordinates(path):
    imgNames = [f for f in os.listdir(path)]
    data = [gdal.Open(os.path.join(path, file )) for file in imgNames]
    width = [data[i].RasterXSize for i in range(len(data))]
    height = [data[i].RasterYSize for i in range(len(data))]
    gt = [data[i].GetGeoTransform() for i in range(len(data))]
    minx = [gt[i][0] for i in range(len(data))]
    miny = [gt[i][3] + width[i]*gt[i][4] + height[i]*gt[i][5] for i in range(len(data))]
    maxx = [gt[i][0] + width[i]*gt[i][1] + height[i]*gt[i][2] for i in range(len(data))]
    maxy = [gt[i][3] for i in range(len(data))] 
    dataset = {'imgName':imgNames, 'minX':minx, 'minY':miny, 'maxX':maxx, 'maxY':maxy}
    coords = pd.DataFrame(data=dataset)
    coords.to_csv(os.path.join(os.path.dirname(noaaPath),'NoaaCoords.csv'))
        
        
    

path = os.path.join(os.path.expanduser('~'),'floodDetection')
noaaPath = os.path.join(path,'Raw Noaa Images')
    
harvey = ['https://ngsstormviewer.blob.core.windows.net/downloads/20170827_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170828a_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170828b_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170829a_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170829b_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170830_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170831a_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170831b_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170901a_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170901b_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170901c_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170902a_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170902b_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170902c_RGB.tar',
         'https://ngsstormviewer.blob.core.windows.net/downloads/20170903a_RGB.tar',
         'https://web.corral.tacc.utexas.edu/nfiedata/Harvey/harvey_hand_float.tif'
         ]

florance = ['https://ngsstormviewer.blob.core.windows.net/downloads/20180915a_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180916a_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180916b_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180917a_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180918a_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180919a_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180919b_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180919c_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180920a_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180920b_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180920c_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180921a_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180921b_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180922a_RGB.tar',
            'https://ngsstormviewer.blob.core.windows.net/downloads/20180922b_RGB.tar']


fetcher(harvey,path)
extractor(path)
coordinates(noaaPath)


