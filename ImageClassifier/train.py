import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="directory with images")
parser.add_argument("gpu",help="include GPU, True or False")
parser.add_argument("arch",help="architecture- vgg16_bn or vgg19_bn")
args = parser.parse_args()
data_dir = args.data_dir
gpu = args.gpu
arch = args.arch


#data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

num_labels=102

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(60),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means,std)
    ])

# TODO: Load the datasets with ImageFolder
train_image_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets,
                                                batch_size=64,
                                                shuffle=True)

test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means,std)
    ])

# TODO: Load the datasets with ImageFolder
test_image_datasets = datasets.ImageFolder(test_dir,transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets,
                                                batch_size=64,
                                                shuffle=True)

valid_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means,std)
    ])

# TODO: Load the datasets with ImageFolder
valid_image_datasets = datasets.ImageFolder(valid_dir,transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets,
                                                batch_size=64,
                                                shuffle=True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def network_vgg16(labels=102,drop=0.5, hidden1=512, hidden2=256,learn_rate=0.0001):
    model = models.vgg16_bn(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
        
    # Hyperparameters for our network
    input_size = 25088

    # Build a feed-forward network
    classifier = nn.Sequential(nn.Linear(input_size, hidden1),
                               nn.ReLU(),
                               nn.Dropout(drop),
                               nn.Linear(hidden1, hidden2),
                               nn.ReLU(),
                               nn.Dropout(drop),
                               nn.Linear(hidden2, labels),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier

    crit = nn.NLLLoss()

    optimize = optim.Adam(
        model.classifier.parameters(),
        learn_rate)

    model.cuda()

    return model, crit, optimize

def network_vgg19(labels=102,drop=0.5, hidden1=512, hidden2=256,learn_rate=0.0001):
    model = models.vgg19_bn(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
        
    # Hyperparameters for our network
    input_size = 25088

    # Build a feed-forward network
    classifier = nn.Sequential(nn.Linear(input_size, hidden1),
                               nn.ReLU(),
                               nn.Dropout(drop),
                               nn.Linear(hidden1, hidden2),
                               nn.ReLU(),
                               nn.Dropout(drop),
                               nn.Linear(hidden2, labels),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier

    crit = nn.NLLLoss()

    optimize = optim.Adam(
        model.classifier.parameters(),
        learn_rate)

    model.cuda()

    return model, crit, optimize

if arch == 'vgg16_bn':
model,critereon,optimizer = network_vgg16(num_labels,drop=0.5,hidden1=1024,hidden2=512,learn_rate=0.0003)

model.to('cuda')
epochs = 10
print_every = 100
steps = 0

print(f'Begin Training...')

for e in range(epochs):
    running_loss = 0
    for _, (images, labels) in enumerate(train_dataloaders):
        steps += 1
        optimizer.zero_grad()
        im = images.to('cuda')
        lbl = labels.to('cuda')
        # Forward and backward passes
        output = model.forward(im)
        loss = critereon(output, lbl)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            acc_v = 0
            loss_v = 0
            #print("Epoch: {}/{}... ".format(e+1, epochs),
            #      "Loss: {:.4f}".format(running_loss/print_every))
            for _,(images_validation,labels_validation) in enumerate(valid_dataloaders):
                optimizer.zero_grad()
                im_v  = images_validation.to('cuda')
                lbl_v = labels_validation.to('cuda')
                with torch.no_grad():
                    out_v  = model.forward(im_v)
                    loss_v = critereon(out_v,lbl_v)
                    p = torch.exp(out_v).data
                    tp = (lbl_v.data == p.max(1)[1])
                    acc_v += tp.type_as(torch.FloatTensor()).mean()
                
            valid_size = len(valid_dataloaders)
            loss_t = running_loss/len(train_dataloaders)
            loss_v = loss_v / valid_size
            acc_v  = acc_v  / valid_size
            running_loss = 0
        
            print(f'Epoch: {e+1}, Train Loss: {loss_t:.3f}, Valid. Loss: {loss_v:.3f}, Valid. Acc: {acc_v:.3f}')
            steps = 0
            model.train()
            
checkpoint = {'state_dict':model.state_dict(),
              'classifier': model.classifier,
              'class_to_idx':train_image_datasets.class_to_idx,
              'opt_state':optimizer.state_dict,
              'num_epochs':epochs
             }

torch.save(checkpoint, 'checkpoint.pth')