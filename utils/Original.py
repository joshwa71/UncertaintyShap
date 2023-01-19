import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import shap

batch_size = 128
num_epochs = 2
device = torch.device('cpu')

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
#
    def entropy(self, x):
        _x = x
        logx = torch.log(_x)
        out = _x * logx
        out = torch.sum(out, 1)
        out = out[:, None]
        return -out
#
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
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
            for i in target:
                if i == 8:
                    target[i] = 1
                if i == 3:
                    target[i] = 0

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output.log(), target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


#train
train_dataset = datasets.MNIST('mnist_data', train=True, transform=transforms.Compose([
                       transforms.ToTensor()
                   ]))
train_idx1 = torch.tensor(train_dataset.targets) == 3
train_idx2 = torch.tensor(train_dataset.targets) == 8

train_indices_3 = train_idx1.nonzero().reshape(-1)
train_indices_8 = train_idx2.nonzero().reshape(-1)

for i in train_indices_3:
    train_dataset.targets[i] = 0
for j in train_indices_8:
    train_dataset.targets[j] = 1

train_mask = train_idx1 | train_idx2
train_indices = train_mask.nonzero().reshape(-1)
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

#test
test_dataset = datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                   ]))
test_idx1 = torch.tensor(test_dataset.targets) == 3
test_idx2 = torch.tensor(test_dataset.targets) == 8

test_indices_3 = test_idx1.nonzero().reshape(-1)
test_indices_8 = test_idx2.nonzero().reshape(-1)

for i in test_indices_3:
    test_dataset.targets[i] = 0
for j in test_indices_8:
    test_dataset.targets[j] = 1

test_mask = test_idx1 | test_idx2
test_indices = test_mask.nonzero().reshape(-1)
test_subset = torch.utils.data.Subset(test_dataset, test_indices)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True)


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

batch = next(iter(test_loader))
images, _ = batch

background = images[:100]
test_images = images[100:104]

e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(test_images)
#print(np.shape(shap_values))

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
#print(np.shape(shap_numpy))
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
#print(np.shape(test_numpy))

shap.image_plot(shap_numpy, -test_numpy)