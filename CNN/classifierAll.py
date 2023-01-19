import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shap
from PIL import Image

# Hyper-parameters
MODE = 'train'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 12
BATCH_SIZE = 512
LEARNING_RATE = 0.001
ROOTDIR = r'C:\Users\joshu\Documents\ArtificialIntelligence\BirdsCNN'

class BirdsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 0]))
        if self.transform:
            image = self.transform(image)
        return (image, y_label)


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv_sec = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.fc_sec = nn.Sequential(
            nn.Linear(16*54*54, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 450)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv_out = self.conv_sec(x)
        flatten_out = conv_out.view(-1, 16*54*54)
        fc_out = self.fc_sec(flatten_out)
        fc_out = self.softmax(fc_out)
        return fc_out


class EntropyClassifier(nn.Module):
    def __init__(self):
        super(EntropyClassifier, self).__init__()
        self.conv_sec = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.fc_sec = nn.Sequential(
            nn.Linear(16*54*54, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2)
        )
        self.softmax = nn.Softmax(dim=1)

    def entropy(self, x):
        _x = x
        logx = torch.log(_x)
        out = _x * logx
        out = torch.sum(out, 1)
        out = out[:, None]
        return -out

    def forward(self, x):
        conv_out = self.conv_sec(x)
        flatten_out = conv_out.view(-1, 16*54*54)
        fc_out = self.fc_sec(flatten_out)
        fc_out = self.softmax(fc_out)
        fc_out = self.entropy(fc_out)
        return fc_out


def train(model, train_loader, optimizer, path):
    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            scores = model(data)
            loss = F.nll_loss(scores.log(), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Train Epoch: ', epoch, 'Loss: ', np.mean(losses))
    torch.save(model.state_dict(), path)

    

#Create data loaders
dataset = BirdsDataset(csv_file='birds.csv', root_dir=ROOTDIR, transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)+1])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)


model = CNNClassifier().to(DEVICE)
entropyModel = EntropyClassifier().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#criterion = nn.CrossEntropyLoss()

if MODE == 'train':
    train(model, train_loader, optimizer, ROOTDIR + r'\model.pth')

if MODE =='test':
    model.load_state_dict(torch.load(ROOTDIR + r'\model.pth'))
    model.eval()
    image = Image.open(ROOTDIR + r'\BushTurkey.jpg')
    convert = transforms.ToTensor()
    image = convert(image).to(DEVICE)
    batch = next(iter(test_loader))
    images, _ = batch

    background = images[0:20].to(DEVICE) 

    test_images = images[21:23].to(DEVICE)

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_images)   
    print(shap_values)

if MODE == 'explain':
    entropyModel.load_state_dict(torch.load(ROOTDIR + r'\model.pth'))
    entropyModel.eval()
    image = Image.open(ROOTDIR + r'\BushTurkey.jpg')
    convert = transforms.ToTensor()
    image = convert(image).to(DEVICE)

    batch = next(iter(test_loader))
    images, targets = batch

    background = images[0:100].to(DEVICE) 

    test_images = images[100:256].to(DEVICE)

    entropies = entropyModel(test_images)
    max_entropy = torch.topk(entropies.flatten(), 5)

    indices = max_entropy.indices + 100

    indices_cpu = indices.to(torch.device('cpu'))
    high_entropy = images[indices_cpu].to(DEVICE)
    high_entropies = entropyModel(high_entropy)

    print(targets[indices_cpu])

    print(high_entropies)

    explainer = shap.DeepExplainer(entropyModel, background)


    shap_values = explainer.shap_values(high_entropy)
    shap_numpy = np.asarray(shap_values)
    
    shap_numpy = [np.swapaxes(np.swapaxes(shap_numpy, 1, -1), 1, 2)]
    test_numpy = np.swapaxes(np.swapaxes(high_entropy.detach().cpu().numpy(), 1, -1), 1, 2)

    shap.image_plot(shap_numpy, test_numpy)
     





