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

options = parsers.parse_args()
data_dir = options.data_dir
save_dir = options.save_dir
top_k = options.top_k
category_names = options.category_names
device = options.device
path_to_image = data_dir

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
checkpoint = torch.load('checkpoint.pth')
checkpoint.keys()
model = models.vgg16(pretrained=True)
classifier = nn.Sequential(
    nn.Linear(25088, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, 102),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier
model.class_to_idx = checkpoint['class_to_idx']
model.load_state_dict(checkpoint['state_dict'])


def process_image(path_to_image):
    image = Image.open(path_to_image)
    width, height = image.size
    if width > height:
        image.resize((width * 256 // height, 256))
    else:
        image.resize((256, height * 256 // width))
    image = image.crop(
        ((image.width - 224) / 2, (image.height - 224) / 2, (image.width - 224) / 2 + 224, (image.height - 224) / 2 + 224))
    np_image = np.array(image)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image


process_image(path_to_image)


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
    processed_image = process_image(image_path)
    processed_image = torch.from_numpy(processed_image).type(torch.FloatTensor)
    processed_image.unsqueeze_(0)
    result = model.forward(processed_image)
    pro = torch.exp(result)
    top_pro, top_labels = pro.topk(5)
    return top_pro, top_labels


predict(path_to_image, model)


def display_img(image_path, model):
    image_to_be_display = process_image(image_path)
    imshow(image_to_be_display, plt.subplot(2, 1, 1))
    probs, classes = predict(image_path, model)
    probs = probs.tolist()[0]
    classes = classes.tolist()[0]
    flower = [i for i in classes]
    flower_cat = []
    for x in flower:
        for key, value in checkpoint['class_to_idx'].items():
            if value == x:
                flower_cat.append(key)
    flower_name = [cat_to_name[i] for i in flower_cat]
    ax = plt.subplot(2,1,2)
    ax.barh(flower_name, probs)
    plt.show()


display_img(path_to_image, model)
