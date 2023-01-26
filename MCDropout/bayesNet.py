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
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x
    
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            print('Dropout!')
            m.train()


def get_monte_carlo_predictions(data_loader, forward_passes, model, n_classes, n_samples):
    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)
        for i, (image, label) in enumerate(data_loader):
            image = image.to(torch.device('cuda'))
            with torch.no_grad():
                output = model(image)
                output = softmax(output) # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.cpu().numpy()))
            print(predictions[np.newaxis, :, :].shape)
        print(dropout_predictions.shape)
        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)
    
    # Calculating mean across multiple MCD forward passes 
    mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes 
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions*np.log(dropout_predictions + epsilon),
                                            axis=-1), axis=0) # shape (n_samples,)
    
    return mean, variance, entropy, mutual_info


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







# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output.log(), target).item() # sum up batch loss
#             pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#     100. * correct / len(test_loader.dataset)))



if __name__ == '__main__':

    mode = 'test'
    path = r"C:\Users\joshu\Documents\ArtificialIntelligence\MetaSHAP\models\MCDropout.pth"
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

    if mode == 'train':
        for epoch in range(1, num_epochs + 1):
            train(meanModel, device, train_loader, optimizer, epoch)
            #test(meanModel, device, test_loader)
        torch.save(meanModel.state_dict(), path)

    if mode == 'test':
        mean, variance, entropy, mutual_info = get_monte_carlo_predictions(test_loader, 10, meanModel, 2, len(test_subset))
        print(mean.shape)
        print(variance.shape)
        print(entropy.shape)
