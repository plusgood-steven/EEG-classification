#%% 
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from dataloader import read_bci_data
from model import EEGNet, DeepConvNet

class BCIDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

#%% 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'n_epochs': 1000,     # Number of epochs.
    'batch_size': 1080,
    'learning_rate': 1e-3,
    # If model has not improved for this many consecutive epochs, stop training.
    'early_stop': 400,
    'save_path': './models/model.ckpt',  # model will be saved here.
    "activation_function": nn.ELU
}

x_train, y_train, x_test, y_test = read_bci_data()

train_dataset, test_dataset = BCIDataset(
    x_train, y_train), BCIDataset(x_test, y_test)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(
    test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

#%%
class AddNoise:
    def __init__(self,sigma):
        self.sigma = sigma
    
    def __call__(self, inputTensor):
        noise = torch.randn_like(inputTensor)
        return inputTensor + (self.sigma ** 0.5) * noise

#%%
# transforms_func = transforms.Compose([
#     transforms.RandomApply(transforms = [AddNoise(0.1)],p=0.5),
#     transforms.RandomApply(transforms = [transforms.RandomCrop((2, 750), padding=(20, 0, 20, 0),
#                           pad_if_needed=False, fill=0, padding_mode='constant')],p=0.5)  
# ])

transforms_func = AddNoise(0.5)

for _,(x,y) in enumerate(train_loader):
    print(x[0][0][0])
    a = transforms_func(x)
    print(a[0][0][0])
#%%
plt.title("Train Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
epochs = np.array([e for e in range(len(x[0][0][0]))])
plt.plot(epochs, np.array(a[0][0][0]), label="AddNoise")
plt.plot(epochs, np.array(x[0][0][0]), label="NoUse")
plt.legend()
plt.show()
# %%
