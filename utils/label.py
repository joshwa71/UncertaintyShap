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
num_epochs = 10
device = torch.device('cuda')


if __name__ == '__main__':

    mode = 'test'
    path = r"C:\Users\joshu\Documents\ArtificialIntelligence\MetaSHAP\models\modelActive17.pth"
    #train
    train_dataset = datasets.MNIST('mnist_data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor()
                    ]))
    

    #traindata
    train_idx1 = torch.tensor(train_dataset.targets) == 1
    pretrain_idx2 = torch.tensor(train_dataset.targets) == 7

    pretrain_dataset = train_dataset

    pretrain_indices_1 = train_idx1.nonzero().reshape(-1)
    pretrain_indices_7 = pretrain_idx2.nonzero().reshape(-1)

    for j in pretrain_indices_7:
        pretrain_dataset.targets[j] = 1
    for i in pretrain_indices_1:
        pretrain_dataset.targets[i] = 0

    normal_indices = np.load('normal.npy') 
    bar_indices = np.load('bar.npy')

    pretrain_mask_7 = pretrain_idx2
    pretrain_mask_1 = train_idx1
    pretrain_indices_7 = pretrain_mask_7.nonzero().reshape(-1)
    pretrain_indices_1 = pretrain_mask_1.nonzero().reshape(-1)
    pretrain_subset = torch.utils.data.Subset(pretrain_dataset, pretrain_indices_7)
    train_subset_7 = torch.utils.data.Subset(pretrain_subset, normal_indices)





    for i in range(1000):
        image = train_subset_7[i][0]
        label = train_subset_7[i][1]
        print(i)
        plt.imshow(image.reshape(28,28), cmap='gray')
        plt.show()



    # for i in range(len(train_subset)):
    # #for i in range(10):
    #     image = train_subset[i][0]
    #     label = train_subset[i][1]
    #     print(label)
    #     plt.imshow(image.reshape(28,28), cmap='gray')
    #     plt.show()
    #     while True:
    #         if keyboard.read_key() == "a":
    #             plt.close()
    #             normal_7s.append(i)
    #             break
    #         if keyboard.read_key() == "l":
    #             plt.close()
    #             bar_7s.append(i)
    #             break
    # print("Normal 7s")
    # print(normal_7s)
    # print("Bar 7s")
    # print(bar_7s)
    # normal_7s = np.asarray(normal_7s)
    # bar_7s = np.asarray(bar_7s)
    # np.save('normal.npy', normal_7s)
    # np.save('bar.npy', bar_7s)


