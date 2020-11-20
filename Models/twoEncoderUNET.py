#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:40:11 2020

@author: abdullahsaid
"""

import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch

encoder = models.resnet50(True)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size=3, padding=1,se = False):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernal_size,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernal_size,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
        )
        if se == True:
            self.se_layer = SELayer(out_channels, 16)
        else:
            self.se_layer = nn.Identity()
    def forward(self, x):
        x = self.layer(x)
        x = self.se_layer(x)
        return x
        

class TwoEncoderUNet(nn.Module):
    def __init__(self, encoder_1, encoder_2, in_channels_1, in_channels_2, 
                 n_classes, decoder_channels=[64,128,256,512,1024], 
                 interpolation = 'nearest', base_width = 64, se = False):
        super(TwoEncoderUNet, self).__init__()
        self.interpolation = interpolation
        #self.n_classes = n_classes
        
        encoder_1_channels = [base_width]+[list(encoder_1.children())[x][-1].conv3.out_channels
                            for x in range(4,8)]
        encoder_2_channels = [base_width]+[list(encoder_1.children())[x][-1].conv3.out_channels
                            for x in range(4,8)]
        
        self.se = se
        #Encoder 1
        
        self.layer_0_1 = nn.Sequential(
            nn.Conv2d(in_channels_1, base_width, kernel_size=7, stride = 2, padding=3),
            nn.BatchNorm2d(base_width),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer_1_1 = encoder_1.layer1
        self.layer_2_1 = encoder_1.layer2
        self.layer_3_1 = encoder_1.layer3
        self.layer_4_1 = encoder_1.layer4
        
        #Encoder 2
        
        self.layer_0_2 = nn.Sequential(
            nn.Conv2d(in_channels_2, base_width, kernel_size=7, stride = 2, padding=3),
            nn.BatchNorm2d(base_width),
            nn.ReLU(True)
        )
        self.layer_1_2 = encoder_2.layer1
        self.layer_2_2 = encoder_2.layer2
        self.layer_3_2 = encoder_2.layer3
        self.layer_4_2 = encoder_2.layer4
        
        #Decoder
        
        self.up_layer0 = DecoderBlock(encoder_1_channels[4]+encoder_2_channels[4]
                                      ,decoder_channels[4], se = se)
        
        self.up_layer1 = DecoderBlock(encoder_1_channels[3]+encoder_2_channels[3]
                                      +decoder_channels[4],decoder_channels[3],
                                      se = se)
        
        self.up_layer2 = DecoderBlock(encoder_1_channels[2]+encoder_2_channels[2]
                                      +decoder_channels[3],decoder_channels[2],
                                      se = se)
        
        self.up_layer3 = DecoderBlock(encoder_1_channels[1]+encoder_2_channels[1]
                                      +decoder_channels[2],decoder_channels[1],
                                      se = se)
        
        self.up_layer4 = DecoderBlock(encoder_1_channels[0]+encoder_2_channels[0]
                                      +decoder_channels[1],decoder_channels[0],
                                      se = se)
        
        #Classifer
        self.segmentation = nn.Sequential(
            nn.Conv2d(decoder_channels[0], n_classes, kernel_size=3, padding=1),
            nn.Sigmoid()
            )

    def forward(self, x, y):
        
        #encoder 1
        x0 = self.layer_0_1(x)
        x1 = self.pool(x0)
        x1 = self.layer_1_1(x1)
        x2 = self.layer_2_1(x1)
        x3 = self.layer_3_1(x2)
        x4 = self.layer_4_1(x3)
        
        #encoder 2
        y0 = self.layer_0_2(y)
        y1 = self.pool(y0)
        y1 = self.layer_1_2(y1)
        y2 = self.layer_2_2(y1)
        y3 = self.layer_3_2(y2)
        y4 = self.layer_4_2(y3)
        
        #decoder
        z = torch.cat([x4,y4], dim=1)
        z = self.up_layer0(z)
        z = F.interpolate(z,scale_factor=2, mode=self.interpolation)
        
        z = torch.cat([z,x3,y3],dim=1)
        z = self.up_layer1(z)
        z = F.interpolate(z,scale_factor=2, mode=self.interpolation)
        
        z = torch.cat([z,x2,y2],dim=1)
        z = self.up_layer2(z)
        z = F.interpolate(z,scale_factor=2, mode=self.interpolation)
        
        z = torch.cat([z,x1,y1],dim=1)
        z = self.up_layer3(z)
        z = F.interpolate(z,scale_factor=2, mode=self.interpolation)
        
        z = torch.cat([z,x0,y0],dim=1)
        z = self.up_layer4(z)
        z = F.interpolate(z,scale_factor=2, mode=self.interpolation)

        z = self.segmentation(z)

        
        return z
