import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import shap

batch_size = 128
num_epochs = 2
device = torch.device('cuda')
path = r"C:\Users\joshu\Documents\ArtificialIntelligence\MetaSHAP\models\model17.pth"

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
        #x = self.entropy(x)
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
    mode = 'test'
    path = r"..\models\model17.pth"
    
    # Train data loading and preprocessing
    train_dataset = datasets.MNIST('mnist_data', train=True, download=True, transform=transforms.ToTensor())
    train_indices_1 = (train_dataset.targets == 1).nonzero().reshape(-1)
    train_indices_7 = (train_dataset.targets == 7).nonzero().reshape(-1)
    train_subset = torch.utils.data.Subset(train_dataset, torch.cat([train_indices_1, train_indices_7]))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    #Test data loading and preprocessing
    test_dataset = datasets.MNIST('mnist_data', train=False, download=True, transform=transforms.ToTensor())
    test_indices_1 = (test_dataset.targets == 1).nonzero().reshape(-1)
    test_indices_7 = (test_dataset.targets == 7).nonzero().reshape(-1)
    test_subset = torch.utils.data.Subset(test_dataset, torch.cat([test_indices_1, test_indices_7]))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    # Model initialization and training
    entropy_model = EntropyNet().to(device)
    mean_model = Net().to(device)
    optimizer = optim.SGD(entropy_model.parameters(), lr=0.01, momentum=0.5)

    if mode == 'train':
        for epoch in range(1, num_epochs + 1):
            train(mean_model, device, train_loader, optimizer, epoch)
            test(mean_model, device, test_loader)
        torch.save(mean_model.state_dict(), path)

    # Model loading and evaluation
    if mode == 'test':
        mean_model.load_state_dict(torch.load(path))
        mean_model.eval()
        entropy_model.load_state_dict(torch.load(path))
        entropy_model.eval()

        batch = next(iter(test_loader))
        images, _ = batch
        background = images[0:100].to(device)
        
        entropies = entropy_model(background)
        mean_entropy = torch.mean(entropies)
        max_entropy = torch.topk(entropies.flatten(), 5)
        indices_cpu = max_entropy.indices.to(torch.device('cpu'))
        test_images = images[indices_cpu].to(device)
        test_entropies = entropy_model(test_images)
        norm_entropies = test_entropies - mean_entropy

        e1 = shap.DeepExplainer(entropy_model, background)
        e2 = shap.DeepExplainer(mean_model, background)
        
        shap_values_entropy = np.asarray(e1.shap_values(test_images))
        shap_values_mean = np.asarray(e2.shap_values(test_images))

        test_images_cpu = test_images.to(torch.device('cpu'))
        shap_entropy_numpy = np.swapaxes(np.swapaxes(shap_values_entropy, 1, -1), 1, 2)
        test_entropy_numpy = np.swapaxes(np.swapaxes(np.asarray(test_images_cpu), 1, -1), 1, 2)
        shap.image_plot(shap_entropy_numpy, -test_entropy_numpy)

        shap_mean_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values_mean]
        test_mean_numpy = np.swapaxes(np.swapaxes(np.asarray(test_images_cpu), 1, -1), 1, 2)
        shap.image_plot(shap_mean_numpy, -test_mean_numpy)