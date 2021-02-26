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
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="checkpoint, without .pth extension")
parser.add_argument("imaget", help="imaget")
parser.add_argument("--gpu",help="GPU availablity, True or False, default is True",type=bool,default=True)
parser.add_argument("--k",help="Top k scores, default is 5", type=int,default=5)
parser.add_argument("--cnames",help="Category names, default is cat_to_name.json",type=str,default='cat_to_name.json')
args = parser.parse_args()
checkpoint = args.checkpoint
imaget = args.imaget


if args.gpu and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

k = args.k
cnames = args.cnames
num_labels = 102

loss_t = 0
acc_t = 0



with open(cnames, 'r') as f:
    cat_to_name = json.load(f)

means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

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

def loader(file):
    '''
    #input is pytorch checkpoint file, and outputs file to load for reuse
    '''
    chkpnt = torch.load(file)
    num_labels = chkpnt['classifier'][6].out_features
    dropout = chkpnt['classifier'][2].p
    hidden1 = chkpnt['classifier'][0].out_features
    hidden2 = chkpnt['classifier'][3].out_features
    model,_,_= network(chkpnt['structure'],num_labels,drop=dropout,hidden1=hidden1,hidden2=hidden2,learn_rate=0.0003)
    model.load_state_dict(chkpnt['state_dict'])
    model.classifier = chkpnt['classifier']
    model.class_to_idx = chkpnt['class_to_idx']
    return model


model2 = loader(f'{checkpoint}.pth')

#print(model2)

image_path=f'{os.getcwd()}/{imaget}.jpg'
print(image_path)

def process_image(image_path):
    im = PIL.Image.open(image_path)
    transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means,std)
    ])
    trans = transform(im)
    return np.array(trans)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #model.to('cpu')
    loaded_model = loader(model)
    im = process_image(image_path)
    # image in torch
    if args.gpu and torch.cuda.is_available():
        imt = torch.from_numpy(im).type(torch.cuda.FloatTensor)
    else:
        imt = torch.from_numpy(im).type(torch.FloatTensor)
    #imt = torch.from_numpy(im).type(torch.cuda.FloatTensor)
    # unsqueeze 
    imt_uns = imt.unsqueeze_(0) # in place
    imt_uns = imt_uns.float()
    # TODO: Implement the code to predict the class from an image file
    loaded_model.eval()
    with torch.no_grad():
        if args.gpu and torch.cuda.is_available():
            output = loaded_model.forward(imt_uns.cuda())
        else:
            output = loaded_model.forward(imt_uns.cpu())
        
        
    prob = torch.exp(output)
    probs, ind = prob.topk(topk)
    
    return prob.topk(topk)

modeln=torch.load(f'{checkpoint}.pth')
flip={}
for i,j in modeln['class_to_idx'].items():
    flip[j]=int(i)

p=predict(image_path,f'{checkpoint}.pth',k)
class_names = [cat_to_name[str(flip[i])] for i in p[1][0].tolist()]
proba = p[0][0].tolist()

for i in range(k):
    print(proba[i],class_names[i])
