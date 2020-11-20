#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:50:54 2020

@author: abdullahsaid
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import gc

def folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

csv = os.path.join(os.path.expanduser('~'),'floodDetection','NoaaCoords.csv')
geo = os.path.join(os.path.expanduser('~'),'floodDetection','GeoJson')
mask = os.path.join(os.path.expanduser('~'),'floodDetection','Mask')

files = [f for f in os.listdir(geo)]
comp = [f[:-4]+'.geojson' for f in os.listdir(mask)]
files = list(set(files)-set(comp))

coords = pd.read_csv(csv)
gdf = {x: gpd.read_file(os.path.join(geo,x)) for x in files}
coords.image = coords.image.str[:-4]+'.geojson'

lat = {x: np.array(coords[['minX','maxX']][coords.image==x])[0] for x
in gdf.keys()}
lon = {x: np.array(coords[['minY','maxY']][coords.image==x])[0] for x
in gdf.keys()}
gc.enable()

def Mask(img):
    fig, ax = plt.subplots(1,1,figsize=(31,31), dpi=399.57)
    ax.set_ylim(lon[img])
    ax.set_xlim(lat[img])
    ax.axis('off')
    gdf[img].plot(ax=ax)
    plt.savefig(os.path.join(mask,img[:-8]+'.png'),bbox_inches='tight',pad_inches=0)
    fig.clear()
    ax.clear()
    plt.close()
    print(img)
    gc.collect()

for i in range(len(files)):
    Mask(files[i])



gc.collect()
files = [f for f in os.listdir(geo)]
comp = [f[:-4]+'.geojson' for f in os.listdir(mask)]
files = list(set(files)-set(comp))