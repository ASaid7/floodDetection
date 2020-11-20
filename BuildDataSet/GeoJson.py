#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:08:36 2020

@author: abdullahsaid
"""
import os
def folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
osm = os.path.join(os.path.expanduser('~'),'floodDetection','OSMJson')
geo = os.path.join(os.path.expanduser('~'),'floodDetection','GeoJson')

folder(geo)

pickup = [f for f in os.listdir(osm)]
dropoff = [f[:-5]+'.geojson' for f in os.listdir(osm)]

for i,j in zip(pickup,dropoff):
  os.system('osmtogeojson '+os.path.join(osm,i)+' > '+os.path.join(geo,j))
