import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tempfile import TemporaryFile

import pandas as pd 
import numpy as np
import shap
import keyboard

batch_size = 128
num_epochs = 30
device = torch.device('cuda')
mode = 'test'
path = r"C:\Users\joshu\Documents\ArtificialIntelligence\MetaSHAP\models\modelActive17.pth"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x


class EntropyNet(nn.Module):
    def __init__(self):
        super(EntropyNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 2),
            nn.Softmax(dim=1)
        )

    def entropy(self, x):
        _x = x
        logx = torch.log(_x)
        out = _x * logx
        out = torch.sum(out, 1)
        out = out[:, None]
        return -out

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        x = self.entropy(x)
        return x



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.log(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output.log(), target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    train_dataset = datasets.MNIST('mnist_data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor()
                    ]))
    
    ## Isolate and relabel 1s and 7s ##
    train_idx1 = torch.tensor(train_dataset.targets) == 1
    pretrain_idx2 = torch.tensor(train_dataset.targets) == 7

    pretrain_dataset = train_dataset

    pretrain_indices_1 = train_idx1.nonzero().reshape(-1)
    pretrain_indices_7 = pretrain_idx2.nonzero().reshape(-1)

    for j in pretrain_indices_7:
        pretrain_dataset.targets[j] = 1
    for i in pretrain_indices_1:
        pretrain_dataset.targets[i] = 0

    pretrain_mask_7 = pretrain_idx2
    pretrain_mask_1 = train_idx1
    pretrain_indices_7 = pretrain_mask_7.nonzero().reshape(-1)
    pretrain_indices_1 = pretrain_mask_1.nonzero().reshape(-1)
    pretrain_subset = torch.utils.data.Subset(pretrain_dataset, pretrain_indices_7)

    ## Separate bar 7s and normal 7s ##
    normal_indices = np.load('normal.npy')
    normal_indices = list(normal_indices)

    ## Remove bad samples ##
    mislabel = [359, 677, 678, 681, 730]
    for i in mislabel:
        del normal_indices[i]

    bar_indices = np.load('bar.npy')
    bar_indices = list(bar_indices)

    ## Get data subsets ## 
    train_subset_1 = torch.utils.data.Subset(pretrain_dataset, pretrain_indices_1[:5000]) #1s
    train_subset_7 = torch.utils.data.Subset(pretrain_subset, normal_indices[:995]) #Normal 7s
    train_subset_7_bar = torch.utils.data.Subset(pretrain_subset, bar_indices) #Bar 7s
    test_subset_1 = torch.utils.data.Subset(pretrain_dataset, pretrain_indices_1[:1000])

    train_dataset = torch.utils.data.ConcatDataset([train_subset_1, train_subset_7])
    test_dataset = torch.utils.data.ConcatDataset([test_subset_1, train_subset_7_bar])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1500, shuffle=True)
    bar_test_loader = torch.utils.data.DataLoader(train_subset_7_bar, batch_size=batch_size, shuffle=True) #Bar only


    ## Initialise model ##
    entropyModel = EntropyNet().to(device)
    meanModel = Net().to(device)
    optimizer = optim.SGD(meanModel.parameters(), lr=0.01, momentum=0.5)

    ## Isolate batch of 1s and normal 7s ##
    batch_normal = next(iter(train_loader))
    images_normal, labels_normal = batch_normal
    background_normal = images_normal[:100].to(device)
    # for i in range(10):
    #     print(labels_normal[i])
    #     plt.imshow(images_normal[i].reshape(28,28), cmap='gray')
    #     plt.show()    

    ## Isolate batch of 1s and bar 7s ##
    batch_test = next(iter(test_loader))
    images_test, labels_test = batch_test
    background_test = images_test[:1500].to(device)  
    # for i in range(10):
    #     print(labels_test[i])
    #     plt.imshow(images_test[i].reshape(28,28), cmap='gray')
    #     plt.show()

    ## Isolate batch of bar 7s ##
    batch_bar = next(iter(bar_test_loader))
    images_bar, labels_bar = batch_bar
    background_bar = images_bar[:100].to(device)    


## Train Classifier ##

    if mode == 'train':
        for epoch in range(1, num_epochs + 1):
            train(meanModel, device, train_loader, optimizer, epoch)
        torch.save(meanModel.state_dict(), path)


## Test classifier ##

    if mode == 'test':
        ## Load Model ##
        meanModel.load_state_dict(torch.load(path))
        meanModel.eval()

        ## Load mean weights to entropy model ##
        entropyModel.load_state_dict(torch.load(path))
        entropyModel.eval()

        ## Get max entropy sample ##
        entropies = entropyModel(background_bar)
        max_entropy = torch.topk(entropies.flatten(), 5)

        ## Print estimates of mean and entropy for max entropy samples ##
        indices_cpu = max_entropy.indices.to(torch.device('cpu'))
        test_images = images_bar[indices_cpu].to(device)
        test_entropies = entropyModel(test_images)
        test_classes = meanModel(test_images)
        print(test_entropies)
        print(test_classes)

        ## Get and display shap values ##
        e1 = shap.DeepExplainer(entropyModel, background_test)
        e2 = shap.DeepExplainer(meanModel, background_test)

        shap_values_entropy = e1.shap_values(test_images)
        shap_values_entropy = np.asarray(shap_values_entropy)

        shap_values_mean = e2.shap_values(test_images)
        shap_values_mean = np.asarray(shap_values_mean)

        test_images_cpu = test_images.to(torch.device('cpu'))

        shap_entropy_numpy = np.swapaxes(np.swapaxes(shap_values_entropy, 1, -1), 1, 2)
        test_entropy_numpy = np.swapaxes(np.swapaxes(np.asarray(test_images_cpu), 1, -1), 1, 2)
        shap.image_plot(shap_entropy_numpy, -test_entropy_numpy)

        shap_mean_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values_mean]
        test_mean_numpy = np.swapaxes(np.swapaxes(np.asarray(test_images_cpu), 1, -1), 1, 2)
        shap.image_plot(shap_mean_numpy, -test_mean_numpy)


















#359, 677, 678, 681, 730







