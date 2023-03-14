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
inference_mode = 'all'
train_mode = 'test'
forward_passes = 50
path = r"..\models\MCDropout.pth"

class Net(nn.Module):
    def __init__(self, forward_passes=20, mode='mean'):
        super(Net, self).__init__()

        self.forward_passes = forward_passes
        self.mode = mode

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
        if self.mode == 'point':
            x = self.conv_layers(x)
            x = x.view(-1, 320)
            x = self.fc_layers(x)

        elif self.mode == 'total_entropy':
            x = self.conv_layers(x)
            x = x.view(-1, 320)
            x = self.fc_layers(x)
            x = self.total_entropy(x)

        elif self.mode == 'ep_entropy':
            ep_entropy = self.get_monte_carlo_predictions(x, x.size()[0], self.forward_passes, 'ep_entropy')
            x = ep_entropy

        elif self.mode == 'al_entropy':
            conv_out = self.conv_layers(x)
            conv_out = conv_out.view(-1, 320)
            fc_out = self.fc_layers(conv_out)
            total_entropy = self.total_entropy(fc_out)

            ep_entropy = self.get_monte_carlo_predictions(x, x.size()[0], self.forward_passes, 'ep_entropy')
            al_entropy = total_entropy - ep_entropy
            x = al_entropy
            print(ep_entropy)
            print(al_entropy)
            print(total_entropy)

        else:
            x = self.get_monte_carlo_predictions(x, x.size()[0], self.forward_passes, self.mode)

        return x
    
    def total_entropy(self, x):
        entropy = -torch.sum(x * torch.log(x), dim=-1)
        return entropy[:,None]

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def get_monte_carlo_predictions(self, x, batch, forward_passes, mode):
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
            

        # Calculating mean across multiple MCD forward passes 
        mean = dropout_predictions.mean(dim=0) # shape (n_samples, n_classes)

        # Calculating entropy across multiple MCD forward passes 
        entropy = -torch.sum(dropout_predictions * torch.log(dropout_predictions), dim=-1)
        entropy = entropy.mean(dim=0)
        
        if mode =='mean':
            return mean
        
        if mode == 'ep_entropy':
            return entropy[:,None]
        

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
            
def create_shap_plots(model, background, test_images, mode):
    #Create  shap values
    explainer = shap.DeepExplainer(model, background)
    shap_values = np.asarray(explainer.shap_values(test_images))
    
    #Create shap plots
    if mode == 'al_entropy' or mode == 'total_entropy' or mode == 'ep_entropy': 
        test_images_cpu = test_images.to(torch.device('cpu'))
        shap_entropy_numpy = np.swapaxes(np.swapaxes(shap_values, 1, -1), 1, 2)
        test_entropy_numpy = np.swapaxes(np.swapaxes(np.asarray(test_images_cpu), 1, -1), 1, 2)
        shap.image_plot(shap_entropy_numpy, -test_entropy_numpy)

    elif mode == 'mean' or mode == 'point':
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
        shap.image_plot(shap_numpy, -test_numpy)


if __name__ == '__main__':
    ##############Create test and train sets with only 1's and 7's########################
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

#################################################################################

    trainPointModel = Net().to(device)
    optimizer = optim.SGD(trainPointModel.parameters(), lr=0.01, momentum=0.5)

    batch = next(iter(test_loader))
    images, _ = batch

    background = images[0:100].to(device)
    test_images = images[0:5]    

    if train_mode == 'train':
        for epoch in range(1, num_epochs + 1):
            train(trainPointModel, device, train_loader, optimizer, epoch)
            #test(meanModel, device, test_loader)
        torch.save(trainPointModel.state_dict(), path)

    if train_mode == 'test':
        if inference_mode == 'point':
            #Initialise model(s)
            pointModel = Net(forward_passes=forward_passes, mode=inference_mode).to(device)
            pointModel.load_state_dict(torch.load(path))
            out = pointModel(test_images.to(device))

            create_shap_plots(pointModel, background, test_images, inference_mode)

        elif inference_mode == 'mean':
            #Initialise model(s)
            meanModel = Net(forward_passes=forward_passes, mode=inference_mode).to(device)
            meanModel.load_state_dict(torch.load(path))
            out = meanModel(test_images.to(device))

            create_shap_plots(meanModel, background, test_images, inference_mode)

        elif inference_mode == 'al_entropy':
            #Initialise model(s)
            alEntropyModel = Net(forward_passes=forward_passes, mode=inference_mode).to(device)
            alEntropyModel.load_state_dict(torch.load(path))

            entropies = alEntropyModel(background)
            mean_entropy = torch.mean(entropies)
            max_entropy = torch.topk(entropies.flatten(), 5)
            indices_cpu = max_entropy.indices.to(torch.device('cpu'))
            test_images = images[indices_cpu].to(device)

            create_shap_plots(alEntropyModel, background, test_images, inference_mode)

        elif inference_mode == 'total_entropy':
            #Initialise model(s)
            totalEntropyModel = Net(forward_passes=forward_passes, mode=inference_mode).to(device)
            totalEntropyModel.load_state_dict(torch.load(path))

            entropies = totalEntropyModel(background)
            mean_entropy = torch.mean(entropies)
            max_entropy = torch.topk(entropies.flatten(), 5)
            indices_cpu = max_entropy.indices.to(torch.device('cpu'))
            test_images = images[indices_cpu].to(device)

            create_shap_plots(totalEntropyModel, background, test_images, inference_mode)

        elif inference_mode == 'all':
            #Initialise model(s)
            totalEntropyModel = Net(forward_passes=forward_passes, mode='total_entropy').to(device)
            totalEntropyModel.load_state_dict(torch.load(path))
            alEntropyModel = Net(forward_passes=forward_passes, mode='al_entropy').to(device)
            alEntropyModel.load_state_dict(torch.load(path))
            epEntropyModel = Net(forward_passes=forward_passes, mode='ep_entropy').to(device)
            epEntropyModel.load_state_dict(torch.load(path))

            #Filter subsample for high entropy
            entropies = totalEntropyModel(background)
            #print(len(entropies))
            mean_entropy = torch.mean(entropies)
            max_entropy = torch.topk(entropies.flatten(), 5)
            indices_cpu = max_entropy.indices.to(torch.device('cpu'))
            test_images = images[indices_cpu].to(device)

            #
            print('breakdown')
            epEntropyModel(test_images[0])

            create_shap_plots(totalEntropyModel, background, test_images, 'total_entropy')
            create_shap_plots(alEntropyModel, background, test_images, 'al_entropy')
            create_shap_plots(epEntropyModel, background, test_images, 'ep_entropy')


        else:
            print("Invalid inference mode selected!")
            