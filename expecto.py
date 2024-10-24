import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import time

from scipy.stats import spearmanr


class ConvNetModel(nn.Module):
    def __init__(self,width, nhistones, nfilters, filtsize, padding, poolsize, n_states_linear1, n_states_linear2, noutputs):
        super(ConvNetModel, self).__init__()
        self.conv1 = nn.Conv1d(nhistones, nfilters, filtsize, padding=padding)
        self.pool = nn.MaxPool1d(poolsize)
        self.fc1 = nn.Linear(int( width * nfilters // poolsize ), n_states_linear1)
        self.fc2 = nn.Linear(n_states_linear1, n_states_linear2)
        self.fc3 = nn.Linear(n_states_linear2, noutputs)
        self.dropout = nn.Dropout(0.8) #0.5

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)


def expectoSetup():

    X_train_path = '/home/vegeta/Downloads/ML4G_Project_1_Data/my_histone_data/X1_train.npy'
    y_train_path = '/home/vegeta/Downloads/ML4G_Project_1_Data/my_histone_data/y1_train.npy'

    X_train = torch.from_numpy(np.load(X_train_path)).float()
    y_train = torch.from_numpy(np.load(y_train_path)).float()


    X_val_path = '/home/vegeta/Downloads/ML4G_Project_1_Data/my_histone_data/X2_val.npy'
    y_val_path = '/home/vegeta/Downloads/ML4G_Project_1_Data/my_histone_data/y2_val.npy'

    X_val = torch.from_numpy(np.load(X_val_path)).float()
    y_val = torch.from_numpy(np.load(y_val_path)).float()

    width = X_train.shape[-1]
    nhistones = 5
    nfilters = 50 #200 
    filtsize = 10 #20
    poolsize = 5
    padding = filtsize // 2
    n_states_linear1 = 100  #100  625 #2000
    n_states_linear2 = 20  #20  125  #1000
    noutputs = 1

    batch_size = 16
    num_epochs = 20


def main():
    pass

