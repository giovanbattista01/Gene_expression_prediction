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


class DeepConvNet(nn.Module):
    def __init__(self,width, nfeatures, filter_list, filtsize_list, padding_list, poolsize_list, n_states_linear1, n_states_linear2, noutputs):
        super(DeepConvNet, self).__init__()
        self.conv1 = nn.Conv1d(nfeatures, filter_list[0], filtsize_list[0], padding=padding_list[0])
        self.conv2 = nn.Conv1d(filter_list[0], filter_list[1], filtsize_list[1], padding=padding_list[1])
        self.pool1 = nn.MaxPool1d(poolsize_list[0])
        self.pool2 = nn.MaxPool1d(poolsize_list[1])
        dim = int(width / (poolsize_list[0] * poolsize_list[1]) * filter_list[1])
        self.fc1 = nn.Linear(dim, n_states_linear1)
        self.fc2 = nn.Linear(n_states_linear1, n_states_linear2)
        self.fc3 = nn.Linear(n_states_linear2, noutputs)
        self.dropout = nn.Dropout(0.8) #0.5

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)




def main():
    pass

