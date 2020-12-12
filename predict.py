import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import time
import copy
import json

import argparse
parsers = argparse.ArgumentParser()
parsers.add_argument('data_dir', action="store")
parsers.add_argument('save_dir', action="store")
parsers.add_argument('--top_k', dest="top_k", default=5)
parsers.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json')
parsers.add_argument('--gpu', action="store_const", dest="device", const="gpu", default='cpu')

arguments = parsers.parse_args()
data_dir = arguments.data_dir
save_dir = arguments.save_dir
top_k = arguments.top_k
category_names = arguments.category_names
device = arguments.device
path_to_image = data_dir

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
checkpoint = torch.load('checkpoint.pth')
checkpoint.keys()
model = models.vgg16(pretrained=True)

model.classifier = nn.Sequential( nn.Linear(25088, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, 102), nn.LogSoftmax(dim=1))
model.class_to_idx = checkpoint['class_to_idx']
model.load_state_dict(checkpoint['state_dict'])


def process_image(path_to_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(path_to_image)
    image=image.resize((image.width*256//image.height,256)) if image.width > image.height else image.resize((256,image.height*256//image.width))
    return ((((np.array(image.crop(((image.width-224)/2, (image.height-224)/2, (image.width-224)/2+224, (image.height-224)/2+224))))/255) - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])).transpose((2, 0, 1))
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax
def predict(image_path, model, topk=5):
    return (torch.exp(model.forward((torch.from_numpy( process_image(image_path)).type(torch.FloatTensor)).unsqueeze_(0)))).topk(5)
def display_img(image_path, model):
    category = []
    flower=[]
    name_list=[]
    imshow(process_image(image_path), plt.subplot(2,1,1))
    probs, classes = predict(image_path, model)
    for i in classes.tolist()[0]:
        flower.append(i)
    for x in flower:
        for category_name, category_value in checkpoint['class_to_idx'].items():
            category.append(category_name) if category_value == x else continue
    for i in category:
        name_list.append(cat_to_name[i])
    ax = plt.subplot(2,1,2)
    ax.barh(name_list, probs.tolist()[0])
    plt.show()

process_image(path_to_image)
predict(path_to_image, model)
display_img(path_to_image, model)
