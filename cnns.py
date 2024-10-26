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

        assert(len(filter_list) == len(poolsize_list))

        self.n_conv_layers = len(filter_list)
        self.conv_layers = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(self.n_conv_layers):
            if i == 0:
                in_channels, out_channels  = nfeatures, filter_list[0]
            else:
                in_channels, out_channels  = filter_list[i-1], filter_list[i]
            padding = padding_list[i]
            kernel = filtsize_list[i]
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel, padding=padding))
            self.pools.append(nn.MaxPool1d(poolsize_list[i]))

            dim = int(width / math.prod(poolsize_list) * filter_list[1])
            self.fc1 = nn.Linear(dim, n_states_linear1)
            self.fc2 = nn.Linear(n_states_linear1, n_states_linear2)
            self.fc3 = nn.Linear(n_states_linear2, noutputs)
            self.dropout = nn.Dropout(0.8)

        

    def forward(self, x):

        for i in range(self.n_conv_layers):
            x = self.conv_layers[i](x)
            x = F.relu(x)
            x = self.pools[i](x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)




def main():
    pass

