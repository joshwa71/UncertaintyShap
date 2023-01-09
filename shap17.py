import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import shap

batch_size = 128
num_epochs = 2
device = torch.device('cuda')

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

    mode = 'test'
    path = r"C:\Users\joshu\Documents\ArtificialIntelligence\MetaSHAP\models\model17.pth"

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('mnist_data', train=True, download=True,
    #                 transform=transforms.Compose([
    #                     transforms.ToTensor()
    #                 ])),
    #     batch_size=batch_size, shuffle=True)

    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
    #                     transforms.ToTensor()
    #                 ])),
    #     batch_size=batch_size, shuffle=True)

    #train
    train_dataset = datasets.MNIST('mnist_data', train=True, transform=transforms.Compose([
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


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    batch = next(iter(test_loader))
    images, _ = batch

    background = images[0:100].to(device)    

    if mode == 'train':
        for epoch in range(1, num_epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
        torch.save(model.state_dict(), path)

    if mode == 'test':
        model.load_state_dict(torch.load(path))
        model.eval()
        entropies = model(background)
        mean_entropy = torch.mean(entropies)
        max_entropy = torch.topk(entropies.flatten(), 5)
        indices_cpu = max_entropy.indices.to(torch.device('cpu'))

        test_images = images[indices_cpu].to(device)
        test_entropies = model(test_images)
        norm_entropies = test_entropies - mean_entropy

        e = shap.DeepExplainer(model, background)
        
        shap_values = e.shap_values(test_images)

        shap_values = np.asarray(shap_values)

        test_images_cpu = test_images.to(torch.device('cpu'))

        shap_numpy = np.swapaxes(np.swapaxes(shap_values, 1, -1), 1, 2)
        test_numpy = np.swapaxes(np.swapaxes(np.asarray(test_images_cpu), 1, -1), 1, 2)
        shap.image_plot(shap_numpy, -test_numpy)


    #print(max_entropy.indices)






    ## Fix Entropy for MNIST 
    ## 3's and 8's, 1's and 7's only datasets
    ## Figure for LSTM NLP
    ##
    ## 
