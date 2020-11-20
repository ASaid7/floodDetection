#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 07:32:38 2020

@author: abdullahsaid
"""

import os
import pandas as pd
import random

path = os.path.join(os.path.expanduser('~'),'floodDetection','whiteList.csv')

df = pd.read_csv(path)
files = df.file
files=list(files)
n=len(files)
s0 = round(n*.6)
s1 = round(s0 + n*.2)
random.shuffle(files)

train = files[:s0]
val = files[s0:s1]
test = files[s1:]

pd.DataFrame(train,columns=['files']).to_csv(os.path.join(os.path.dirname(path),'train.csv'))
pd.DataFrame(val,columns=['files']).to_csv(os.path.join(os.path.dirname(path),'val.csv'))
pd.DataFrame(test,columns=['files']).to_csv(os.path.join(os.path.dirname(path),'test.csv'))

