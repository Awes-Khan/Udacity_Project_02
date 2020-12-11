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
parser.add_argument('--hidden_units', action="store", dest="hidden_units", default=500)
parser.add_argument('--epochs', action="store", dest="epochs", default=3)
parser.add_argument('--gpu', action="store_const", dest="device", const="gpu", default='cpu')

results = parser.parse_args()
data_dir = results.data_dir
save_dir = results.save_dir
arch = results.arch
lr = results.learning_rate
hidden_units = results.hidden_units
epochs = results.epochs
device = results.device

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

model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)


def is_valid(model, testloader, criterion, device):
    loss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return loss, accuracy


def my_nn(model, trainloader, testloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0
    if device == 'gpu':
        device = 'cuda'
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = is_valid(model, testloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e + 1, epochs), end=" ")
                print("Training Loss: {:.3f}.. ".format(running_loss / print_every), end=" ")
                print("Test Loss: {:.3f}.. ".format(test_loss / len(testloader)), end=" ")
                print("Test Accuracy: {:.3f}".format(accuracy / len(testloader)))
                running_loss = 0
                model.train()


my_nn(model, dataloaders[0], dataloaders[1], epochs, 40, criterion, optimizer, device)

# Done: Do validation on the test set
def print_accuracy_of_my_nn(testloader, device):
    positive = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            if device == 'gpu':
                device = 'cuda'
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            positive += (predicted == labels).sum().item()

    print('Accuracy: %d %%' % (100 * positive / total))
print_accuracy_of_my_nn(dataloaders[0], 'gpu')
model.class_to_idx = image_datasets[0].class_to_idx
model.cpu()
torch.save({'arch':'vgg16','state_dict':model.state_dict(),'class_to_idx':model.class_to_idx},'checkpoint.pth')