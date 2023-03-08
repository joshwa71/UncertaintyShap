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
        return mean

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



if __name__ == '__main__':

    mode = 'test'
    path = r"..\models\MCDropout.pth"
    #train
    train_dataset = datasets.MNIST('mnist_data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor()
                    ]))
    train_idx1 = torch.tensor(train_dataset.targets) == 1
    train_idx2 = torch.tensor(train_dataset.targets) == 7

    train_indices_1 = train_idx1.nonzero().reshape(-1)
    train_indices_7 = train_idx2.nonzero().reshape(-1)

    for i in train_indices_1:
        train_dataset.targets[i] = 0
    for j in train_indices_7:
        train_dataset.targets[j] = 1

    train_mask = train_idx1 | train_idx2
    train_indices = train_mask.nonzero().reshape(-1)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    #test
    test_dataset = datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                        transforms.ToTensor()
                    ]))
    test_idx1 = torch.tensor(test_dataset.targets) == 1
    test_idx2 = torch.tensor(test_dataset.targets) == 7

    test_indices_1 = test_idx1.nonzero().reshape(-1)
    test_indices_7 = test_idx2.nonzero().reshape(-1)

    for i in test_indices_1:
        test_dataset.targets[i] = 0
    for j in test_indices_7:
        test_dataset.targets[j] = 1

    test_mask = test_idx1 | test_idx2
    test_indices = test_mask.nonzero().reshape(-1)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    meanModel = Net().to(device)
    optimizer = optim.SGD(meanModel.parameters(), lr=0.01, momentum=0.5)

    batch = next(iter(test_loader))
    images, _ = batch

    background = images[0:100].to(device)
    test_images = images[0:2]    

    if mode == 'train':
        for epoch in range(1, num_epochs + 1):
            train(meanModel, device, train_loader, optimizer, epoch)
            #test(meanModel, device, test_loader)
        torch.save(meanModel.state_dict(), path)

    if mode == 'test':
        meanModel.load_state_dict(torch.load(path))
        # mean = meanModel(images[0].to(device))
        # print(mean)
        explainer = shap.DeepExplainer(meanModel, background)
        shap_values = explainer.shap_values(test_images)
        
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
        shap.image_plot(shap_numpy, -test_numpy)
