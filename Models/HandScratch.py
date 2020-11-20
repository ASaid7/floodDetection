#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:04:57 2020

@author: abdullahsaid
"""

import torch
from torch.utils.data import Dataset, DataLoader,RandomSampler
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd
import os
import random
import segmentation_models_pytorch as smp
import numpy as np
from tqdm import tqdm, tnrange
import time
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import lovasz_losses as L


path = os.path.join(os.path.expanduser('~'),'floodDetection','dataSet')

with open(os.path.join(path,'norm.npy'), 'rb') as f:
    mean = np.load(f)
    std = np.load(f)

#imagenet weights used if hand is not included since the model is initlized with image net weights
#when hand is included since four channels are used the models first layer is randoly initilized 
imgNetMean = [0.485, 0.456, 0.406]
imgNetStd = [0.229, 0.224, 0.225]

trainDS = os.path.join(os.path.dirname(path),'trainFinal3.csv')
valDS = os.path.join(os.path.dirname(path),'valFinal3.csv')

class harveyDataset(Dataset):
    '''
    inputs:
    
        csvFile: is the dataset with file names
        hand: boolean if hand value should be included
        phase: if train or validation then apply augmentation otherwise no augmenation is used
        transform: augmentation that will be applied
        p: probability of applying a random augmentation
    
    output:
        
        Iterable dataset for a dataloader
    '''
    def __init__(self, csvFile, hand, phase, transform = None, p=.5):
        self.data = pd.read_csv(csvFile)
        self.phase = phase
        self.hand = hand
        self.transform = transform
        self.p = p

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(path,self.data.loc[idx,'file']+'.jpg'))
        if self.phase in ('train','val'):
            seg = Image.open(os.path.join(path,self.data.loc[idx,'file']+'.png'))
            if self.hand == True:
                hnd = Image.open(os.path.join(path,self.data.loc[idx,'file']+'.tif'))
                if self.transform:
                    imageHand, label = self.transform(image=img,hand=hnd,
                                                        seg=seg,p=self.p)
                    return imageHand, label*255
                else:
                    image = TF.to_tensor(img)
                    hand = TF.to_tensor(hnd)
                    label = TF.to_tensor(seg)
                    imageHand = torch.cat([image,hand],dim=0)
                    imageHand = TF.normalize(imageHand, mean=mean, std = std)
                    return imageHand, label*255
            else:
                if self.transform:
                    image, label = self.transform(image=img,hand=None,
                                                        seg=seg,p=self.p)
                    return image, label*255
                else:
                    image = TF.to_tensor(img)
                    label = TF.to_tensor(seg)
                    image = TF.normalize(image, mean=imgNetMean, std = imgNetStd)
                    return image, label*255
        else:
            image = TF.to_tensor(img)
            image = TF.normalize(image, mean=imgNetMean, std = imgNetStd)
            return image, self.data.loc[idx,'file']


#augment function to insure that the same augmentation is used on labels and hand
def augmentation(image, seg, hand=None, p=.5):
    if hand:
        if random.random() < p:
            angle = random.randint(-45,45)
            image = TF.rotate(image,angle)
            hand = TF.rotate(hand,angle,fill=0.) 
            seg = TF.rotate(seg,angle,fill=2) #if rotation leaves edges black fill value with 2 to ignore
        if random.random() < p:
            image = TF.hflip(image)
            hand = TF.hflip(hand)
            seg = TF.hflip(seg)
        if random.random() < p:
            image = TF.vflip(image)
            hand = TF.vflip(hand)
            seg = TF.vflip(seg)
        if random.random() < p:
            image = TF.adjust_saturation(image,random.uniform(.5,3)) #only applicable to image
        if random.random() < p:
            image = TF.adjust_contrast(image,random.uniform(.5,3)) #only applicable to image
        
        
        image = TF.to_tensor(image)
        hand = TF.to_tensor(hand)
        segmentation = TF.to_tensor(seg)
        imageHand = torch.cat([image,hand],dim=0)
        imageHand = TF.normalize(imageHand, mean=mean, std = std)
        return imageHand, segmentation
    else:
        if random.random() < p:
            angle = random.randint(-45,45)
            image = TF.rotate(image,angle)
            seg = TF.rotate(seg,angle,fill=2)
        if random.random() < p:
            image = TF.hflip(image)
            seg = TF.hflip(seg)
        if random.random() < p:
            image = TF.vflip(image)
            seg = TF.vflip(seg)
        if random.random() < p:
            image = TF.adjust_saturation(image,random.uniform(.5,3))
        if random.random() < p:
            image = TF.adjust_contrast(image,random.uniform(.5,3))

        image = TF.to_tensor(image)
        segmentation = TF.to_tensor(seg)
        image = TF.normalize(image,mean=imgNetMean, std=imgNetStd)
        return image, segmentation

#training procedure, with a annealing cosine scheduler for learning rate
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        iters = len(dataloaders['train'])
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for i, (inputs, labels) in zip(tqdm(range(len(dataloaders[phase]))),dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                if phase == 'train':
                    scheduler.step(epoch + i / iters)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs)
                    pred = (nn.Sigmoid()(output)>.5).long()
                    loss = criterion(output, labels, ignore=2)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics 
                running_loss += loss.item() * inputs.size(0)
                running_corrects += L.iou_binary(pred,labels,ignore=2,per_image=False
                                                 ) * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Overall Loss: {:.4f} IoU: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            log=open(os.path.join(path,'handEpoch.txt'),'a')
            log.writelines('{} Hand Overall Loss: {:.4f} No Hand IoU: {:.4f}\n\n'.format(
                phase, epoch_loss, epoch_acc))
            log.close()
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val IoU: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train=harveyDataset(trainDS, hand=True, phase='train',transform=augmentation,p=.5)
val=harveyDataset(valDS, hand=True, phase='val',transform=augmentation,p=.5)

dataDict = {'train': train,
            'val': val}

sampler = {'train': RandomSampler(train, replacement=False),
           'val': RandomSampler(val, replacement=False)}


dataloaders = {x: DataLoader(dataDict[x], batch_size=4, sampler=sampler[x],
                             num_workers=0, pin_memory=True) for x in ['train','val']}

dataset_sizes = {x: len(sampler[x]) for x in ['train', 'val']}


#Since we are using BCEWithLogitsLoss Sigmoid activation is done inside that function
#More stable than using BCLoss function

model = smp.Unet('efficientnet-b2',in_channels=4,encoder_weights='imagenet',
                 classes=1, activation=None, encoder_depth=5,
                 decoder_channels = (1024, 512, 256, 128, 64))
#freeze the first layer to learn some repersentation since the other layers have learned weights 
ct = 0
for child in model.encoder.children():
    ct += 1
    if ct > 1:
        for param in child.parameters():
            param.requires_grad = False

model = model.to(device)

criterion = L.binary_xloss

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

#SGD with warm resets after 200 iters with a factor of 2 for growth in iter size before next rest
exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,200,2)

model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=5)

modelPath = os.path.join('/','scratch','asaid8')

torch.save(model.state_dict(), os.path.join(os.path.dirname(path),'initalHandWeights.pt'))

#unfreeze entire model
for child in model.encoder.children():
    for param in child.parameters():
        param.requires_grad = True

model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=25)

torch.save(model.state_dict(), os.path.join(os.path.dirname(path),'B2HandScratch.pt'))


