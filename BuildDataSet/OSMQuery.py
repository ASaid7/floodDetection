#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:13:43 2020

@author: abdullahsaid
"""

import requests
import json
import os
import pandas as pd

def folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

path = os.path.join(os.path.expanduser('~'),'floodDetection','OSMJson')
folder(path)

csv = os.path.join(os.path.dirname(path),'NoaaCoords.csv')
coordsMain = pd.read_csv(csv)
coordsMain['imgName'] = coordsMain['imgName'].str[:-4]
coordsMain = coordsMain.set_index('imgName')
coordsMain = coordsMain.rename(columns={'maxY':'north',
                                'minY':'south',
                                'maxX':'east',
                                'minX':'west'})
coordsMain=coordsMain.to_dict('index')
data = {}

for i in coordsMain.keys():
    query = """
    
    [out:json][timeout:25];
    
    (
    
      node["natural"="water"]({0}, {1}, {2}, {3});
      way["natural"="water"]({0}, {1}, {2}, {3});
      relation["natural"="water"]({0}, {1}, {2}, {3});
      
      node["basin"="retention"]({0}, {1}, {2}, {3});
      way["basin"="retention"]({0}, {1}, {2}, {3});
      relation["basin"="retention"]({0}, {1}, {2}, {3});
      
      node["waterway"]({0}, {1}, {2}, {3});
      way["waterway"]({0}, {1}, {2}, {3});
      relation["waterway"]({0}, {1}, {2}, {3});
    
    );
    
    out;
    
    >;
    
    out skel qt;
    
    """.format(coordsMain[i]['south'],coordsMain[i]['west'],coordsMain[i]['north'],coordsMain[i]['east'])
    
    url = "http://overpass-api.de/api/interpreter"
    response = requests.get(url, params={'data': query})
    try:
        data[i] = response.json()
    except:
        continue
    
    
for i in data.keys():
    with open(os.path.join(path,i+'.json'), 'w') as fp:
        json.dump(data[i], fp)