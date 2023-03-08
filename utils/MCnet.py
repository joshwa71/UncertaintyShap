import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import sys
import shap

batch_size = 128
num_epochs = 5
device = torch.device('cuda')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.forward_passes = 10

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(p=0.3),        
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(p=0.3),        
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x = self.conv_layers(x)
        # x = x.view(-1, 320)
        # x = self.fc_layers(x)
        x = self.get_monte_carlo_predictions(x, x.size()[0], self.forward_passes)
        return x
    
    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def get_monte_carlo_predictions(self, x, batch, forward_passes):
        n_classes = 2
        n_samples = batch
        dropout_predictions = torch.empty((0, n_samples, n_classes), device=device)
        for i in range(forward_passes):
            predictions = torch.empty((0, n_classes), device=device)
            self.enable_dropout()
            conv_out = self.conv_layers(x)
            reshape_out = conv_out.view(-1, 320)
            predictions = self.fc_layers(reshape_out)
            dropout_predictions = torch.cat((dropout_predictions, predictions.unsqueeze(0)), dim=0)
            #print(dropout_predictions.size())

        # Calculating mean across multiple MCD forward passes 
        mean = dropout_predictions.mean(dim=0) # shape (n_samples, n_classes)

        # Calculating variance across multiple MCD forward passes 
        variance = dropout_predictions.var(dim=0) # shape (n_samples, n_classes)

        epsilon = torch.finfo(torch.float32).eps
        # Calculating entropy across multiple MCD forward passes 
        entropy = -torch.sum(mean * torch.log(mean + epsilon), dim=-1) # shape (n_samples,)

        # Calculating mutual information across multiple MCD forward passes 
        mutual_info = entropy - torch.mean(torch.sum(-dropout_predictions*torch.log(dropout_predictions + epsilon),
                                                dim=-1), dim=0) # shape (n_samples,)
        #return entropy
        return mean, variance, entropy, mutual_info