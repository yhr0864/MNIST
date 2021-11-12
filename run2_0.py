import torch
import cv2
import random
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torchvision.transforms import transforms as T

from model.mydataset import Data 
from model.CNN import CNN

from datatxt.datatxt import *


#generate txt
Datatxt(r'C:\Users\Myth\Desktop\data\input\train', r'C:\Users\Myth\Desktop\data\output\train').generateTxt()

#settings
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
epochs = 2

datapath = r'G:\PYTHON\netwerk\MNIST\Image\imgs_train\train.txt'


train_data=Data(datatxt=datapath, transform=T.ToTensor())

len_train = int(0.8 * len(train_data))
len_val = len(train_data) - len_train

train_set, val_set = torch.utils.data.random_split(train_data, [len_train, len_val])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

model = CNN().to(device)

optimizer = Adam(model.parameters(), lr=1e-5)


# Training
def train(epoch):
    with tqdm(total=len(train_loader)) as train_bar:
            for i, data in enumerate(train_loader):
                image = data[0].to(device)
                label = data[1].to(device)

                optimizer.zero_grad()

                pred = model(image)

                loss = F.nll_loss(pred, label)

                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print("train epoch: {}, iter: {}, loss: {}".format(epoch, i, loss.item()))

# Validation                    
def valid():
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as val_bar:
            for i, data in enumerate(val_loader):
                image = data[0].to(device)
                label = data[1].to(device)

                pred = model(image) # batch_size * 10

                total_loss += F.nll_loss(pred, label, reduction='sum').item()

                output = pred.argmax(dim=1) # batch_size * 1

                correct += output.eq(label.view_as(output)).sum().item()


    total_loss /= len_val
    acc = correct / len_val * 100
    print("test loss: {}, accuracy: {}".format(total_loss, acc))


def run():
    for epoch in range(epochs):    
        train(epoch)
        valid()
    
    torch.save(model.state_dict(), 'mnist_cnn.pt')
    
if __name__ == '__main__':
    run()
