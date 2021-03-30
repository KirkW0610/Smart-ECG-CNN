from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

from ECGDataset import ECGDataset

import torchvision
from torchvision import transforms, utils

import numpy as np
import matplotlib.pyplot as plt


import warnings


warnings.filterwarnings("ignore")
plt.ion()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare the Data
dataset = ECGDataset(csv_file='ECGDataset.csv', root_dir='ECG_Data', transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [150, 44])
train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=10, shuffle=True)


# Build the model CNN

# class LightingCNN(pl.LightningModule):

class NeuralNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, (5, 5))
        self.conv2 = nn.Conv2d(8, 16, (5, 5))
        self.conv3 = nn.Conv2d(16, 32, (4, 4))

        # linear, fully connected, and dense are the same
        self.fc1 = nn.Linear(in_features=77 * 57 * 32, out_features=2239)
        self.out = nn.Linear(in_features=2239, out_features=3)

    # forward pass method

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=(2, 2), stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=(2, 2), stride=2)

        t = F.relu(self.conv3(t))
        t = F.max_pool2d(t, kernel_size=(2, 2), stride=2)

        t = F.relu(self.fc1(t.reshape(-1, 77 * 57 * 32)))
        t = self.out(t)

        return t

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.cross_entropy_loss(logits, labels)
        self.log('train_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        logits = self.forward(images)
        loss = self.cross_entropy_loss(logits, labels)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=.005)
        return optimizer

    def train_dataloader(self):
        return DataLoader(dataset=train_set, batch_size=10, shuffle=True)

    def test_dataloader(self):
        return DataLoader(dataset=test_set, batch_size=10, shuffle=True)


network = NeuralNetwork()
print(network)

trainer = pl.Trainer()
trainer.fit(network)

'''''
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels.sum().item())

network = NeuralNetwork()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
# optimizer = optim.Adam(network.parameters(), lr=.005)



for epoch in range(11):

    total_loss = 0
    total_correct = 0

    for batch in train_loader:
        images, labels = batch

        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # back propagation
        optimizer.step()  # Update weights

        total_loss += loss.item()
        total_correct = get_num_correct(preds, labels)

        print("epoch:", epoch, "total correct:", total_correct, "total loss:", total_loss)

    # confusion matrix, which will describe the performance of our model,
    
    '''''
