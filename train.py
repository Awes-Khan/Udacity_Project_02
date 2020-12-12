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

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action="store")
parser.add_argument('--save_dir', action="store", dest="save_dir", default='checkpoint')
parser.add_argument('--arch', action="store", dest="arch", default='vgg16')
parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", default=512)
parser.add_argument('--epochs', action="store", dest="epochs", default=3)
parser.add_argument('--gpu', action="store_const", dest="device", const="gpu", default='cpu')

arguments = parser.parse_args()
data_dir = arguments.data_dir
save_dir = arguments.save_dir
arch = arguments.arch
lr = arguments.learning_rate
hidden_units_count = arguments.hidden_units
epochs = arguments.epochs
device = arguments.device
model = models.vgg13(pretrained=True) if arch == 'vgg13' else models.vgg13(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = [

                 transforms.Compose([
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ]),
                transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ]),
                transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ])
                ]

image_datasets = [
                    datasets.ImageFolder(train_dir, transform=data_transforms[0]),
                    datasets.ImageFolder(valid_dir, transform=data_transforms[1]),
                    datasets.ImageFolder(test_dir, transform=data_transforms[2])
                 ]

dataloaders = [
                torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
                torch.utils.data.DataLoader(image_datasets[1], batch_size=64),
                torch.utils.data.DataLoader(image_datasets[2], batch_size=64)
              ]


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model.classifier = nn.Sequential( nn.Linear(25088, hidden_units_count), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(hidden_units_count, 102), nn.LogSoftmax(dim=1))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

def is_valid(model, loader, criterion, device):
    loss = 0
    accuracy = 0
    for (img, labels) in zip(loader):
        loss += criterion(model.forward(img.to(device)), labels.to(device)).item()
        accuracy += (labels.to(device).data == torch.exp(model.forward(img.to(device))).max(dim=1)[1]).type(torch.FloatTensor).mean()
    return loss, accuracy

def my_nn(model, trainloader, testloader, epochs, criterion, optimizer, device):
    device = 'cuda' if device == 'gpu' else 'cpu'
    model.to(device)
    epoch_count=0
    while(epoch_count<int(epochs)):
        current_loss = 0
        for iterator,(input_count, label_name) in enumerate(trainloader):
            optimizer.zero_grad()
            loss = criterion(model.forward(input_count.to(device)), label_name.to(device))
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

# Done: Do validation on the test set
def check_accuracy_of_my_nn(testloader, device):
    positive = 0
    total = 0
    with torch.no_grad():
        for iterator,(image, label_name) in enumerate(testloader):
            device = 'cuda' if device == 'gpu' else 'cpu'
            variable, predicted = torch.max(model(image.to(device)).data, 1)
            total += label_name.to(device).size(0)
            positive += (predicted == label_name).sum().item()
    print('Accuracy: %d %%' % (100 * positive / total))

my_nn(model, dataloaders[0], dataloaders[1], epochs, criterion, optimizer, device)
check_accuracy_of_my_nn(dataloaders[0], 'gpu')
model.class_to_idx = image_datasets[0].class_to_idx
torch.save({'arch':'vgg16', 'state_dict':model.state_dict(), 'class_to_idx':model.class_to_idx}, 'checkpoint.pth')