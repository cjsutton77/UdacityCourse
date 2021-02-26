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

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="checkpoint")
parser.add_argument("imaget", help="imaget")
args = parser.parse_args()
checkpoint = args.checkpoint
imaget = args.imaget

data_dir = 'flowers'
num_labels = 102

loss_t = 0
acc_t = 0



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
#model.eval()
'''
with torch.no_grad():
    for _, (images,labels) in enumerate(test_dataloaders):
        img,lbl = images.to('cuda'),labels.to('cuda')
        output = model.forward(img)
        loss = critereon(output, lbl)
        running_loss += loss.item()

        p = torch.exp(output).data
        tp = (lbl.data == p.max(1)[1])
        acc_t += tp.type_as(torch.FloatTensor()).mean()
        
    print(f'Test loss: {loss/len(test_dataloaders):.3f}, Test Acc: {acc_t/len(test_dataloaders):.3f}')
    
checkpoint = {'state_dict':model.state_dict(),
              'classifier': model.classifier,
              'class_to_idx':train_image_datasets.class_to_idx,
              'opt_state':optimizer.state_dict,
              'num_epochs':epochs
             }

torch.save(checkpoint, 'checkpoint.pth')
'''
def network(labels=102,drop=0.5, hidden1=512, hidden2=256,learn_rate=0.0001):
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

model,critereon,optimizer = network(num_labels,drop=0.5,hidden1=1024,hidden2=512,learn_rate=0.0003)

def loader(file):
    chkpnt = torch.load(file)
    model,_,_= network(num_labels,drop=0.5,hidden1=1024,hidden2=512,learn_rate=0.0003)
    model.load_state_dict(chkpnt['state_dict'])
    model.classifier = chkpnt['classifier']
    model.class_to_idx = chkpnt['class_to_idx']
    
    return model


model2 = loader(f'{checkpoint}.pth')
#print(model2)
image_path=f'/home/workspace/ImageClassifier/flowers/test/{imaget}.jpg'
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
    imt = torch.from_numpy(im).type(torch.cuda.FloatTensor)
    # unsqueeze 
    imt_uns = imt.unsqueeze_(0) # in place
    imt_uns = imt_uns.float()
    # TODO: Implement the code to predict the class from an image file
    loaded_model.eval()
    with torch.no_grad():
        output = loaded_model.forward(imt_uns.cuda())
    prob = torch.exp(output)
    probs, ind = prob.topk(topk)
    
    return prob.topk(topk)
    
p=predict(image_path,f'{checkpoint}.pth',5)
class_names = [cat_to_name[str(i+1)] for i in p[1][0].tolist()]
proba = p[0][0].tolist()

for i in range(5):
    print(proba[i],class_names[i])
