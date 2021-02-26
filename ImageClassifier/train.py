import argparse
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL
import seaborn as sns
import os

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="directory with images")
parser.add_argument("--arch", help="architecture, argument is vgg or dense, default is vgg.",default='vgg')
parser.add_argument("--gpu" ,help="gpu available, argument is True or False", type=bool,default=False)
parser.add_argument("--hl1" ,help="hidden layer 1, default is 512",type=int,default=512)
parser.add_argument("--hl2" ,help="hidden layer 2, default is 256",type=int,default=256)
parser.add_argument("--lr"  ,help="learning rate, default is .0003",type=float,default=0.0003)
parser.add_argument("--e"   ,help="number of epochs, default is 10",type=int,default=10)
parser.add_argument("--drop",help="dropout, default is 0.5",type=float,default=0.5)
args = parser.parse_args()


hl1 = args.hl1
hl2 = args.hl2
lr =  args.lr
drop = args.drop
structure = args.arch
data_dir = args.data_dir
e = args.e
if args.gpu and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Device is {device}')
print(f'Architechture is {structure}')
print(f'Hidden layer 1 is {hl1}')
print(f'Hidden layer 2 is {hl2}')
print(f'Learning rate is {lr}')
print(f'Dropout is {drop}')
print(f'Training epochs is {e}')

input("Please press enter to continue")
    
    
#data_dir = f'{os.getcwd()}'
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

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
    
def network(arch,labels=102,drop=0.5, hidden1=512, hidden2=256,learn_rate=0.0001):
    '''
    input:
    labels (102 categories)
    drop (default 0.5)
    hidden layers 1 & 2
    learning rate
    
    outputs model, critereon and optimizer
    '''
    if arch == 'vgg':
        model = models.vgg19(pretrained=True)
        input_size = 25088
    else:
        model = models.densenet121(pretrained=True)
        input_size = 1024
    for p in model.parameters():
        p.requires_grad = False
        
    # Hyperparameters for our network

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

    if args.gpu and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    return model, crit, optimize



model,critereon,optimizer = network(structure,num_labels,drop=drop,hidden1=hl1,hidden2=hl2,learn_rate=lr)

model.to(device)
epochs = e
print_every = 100
steps = 0

print(f'Begin Training...')

for e in range(epochs):
    running_loss = 0
    for _, (images, labels) in enumerate(train_dataloaders):
        steps += 1
        optimizer.zero_grad()
        im = images.to(device)
        lbl = labels.to(device)
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
                im_v  = images_validation.to(device)
                lbl_v = labels_validation.to(device)
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
              'structure': structure,
              'class_to_idx':train_image_datasets.class_to_idx,
              'opt_state':optimizer.state_dict,
              'num_epochs':epochs
             }

torch.save(checkpoint, 'checkpoint.pth')