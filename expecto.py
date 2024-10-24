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

"""def Spearman(v1, v2):
    rank1 = torch.argsort(torch.argsort(v1))
    rank2 = torch.argsort(torch.argsort(v2))

    d = rank1 - rank2
    d_norm = torch.sum(d.float() ** 2)

    n = len(rank1)
    return 1 - (6 * d_norm) / (n * (n**2 - 1))"""


class RankNet1DLoss(nn.Module):
    def __init__(self, model = None, l2_lambda=0):
        super(RankNet1DLoss, self).__init__()
        self.l2_lambda = l2_lambda
        self.model = model
    
    def forward(self, y_pred, y_true):
        """
        y_pred: Tensor of shape (batch_size,)
            The predicted scores for each item.
        y_true: Tensor of shape (batch_size,)
            The ground truth scores for each item.
        """
        # Ensure the predictions and targets are 1D tensors
        assert y_pred.dim() == 1 and y_true.dim() == 1, "Input tensors must be 1D"
        
        # Get the pairwise differences for the ground truth and predictions
        diff_true = y_true.unsqueeze(0) - y_true.unsqueeze(1)  # Shape: (batch_size, batch_size)
        diff_pred = y_pred.unsqueeze(0) - y_pred.unsqueeze(1)  # Shape: (batch_size, batch_size)
        
        # Only consider pairs where the true ranks are different
        mask = (diff_true > 0).float()  # Shape: (batch_size, batch_size)
        
        # Apply the sigmoid function to the predicted differences
        P_ij = torch.sigmoid(diff_pred).clamp(1e-7, 1 - 1e-7)  # Pairwise probabilities (batch_size, batch_size)
        
        # Binary cross-entropy loss
        bce_loss = mask * torch.log(P_ij + 1e-7) + (1 - mask) * torch.log(1 - P_ij + 1e-7)
        
        # We only sum over valid pairs (where the mask is 1)
        loss = -torch.sum(bce_loss * mask) / ( torch.sum(mask) + 1e-7 )  # Normalize by the number of valid pairs

        if self.model != None and self.l2_lambda > 0:

            l2_reg = 0.0
            for param in self.model.parameters():
                l2_reg += torch.norm(param, 2)

            loss = loss + self.l2_lambda * l2_reg

        return loss



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


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        
        self.X = X
        self.y = y 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        return sample, label

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

model = ConvNetModel(width,nhistones, nfilters, filtsize, padding, poolsize, n_states_linear1, n_states_linear2, noutputs)
#model.load_state_dict(torch.load(weights_path))

batch_size = 16
num_epochs = 20
learning_rate = 0.001


train_dataset = SimpleDataset(X_train, y_train)
val_dataset = SimpleDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


def train_val (train_loader, val_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    #criterion = nn.MSELoss(reduction='mean')
    criterion = RankNet1DLoss(model=model, l2_lambda=0.05)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_spearman = 0

    for epoch in range(num_epochs):
        print("epoch : "+str(epoch))

        print("training...")
        model.train()

        train_loss = 0
        train_prediction = torch.Tensor([]).to(device)
        train_y = torch.Tensor([]).to(device)

        for X_batch, y_batch in train_loader:

            if torch.isnan(X_batch).any():
                continue

            X_batch,y_batch = X_batch.to(device), y_batch.to(device)

            f_X = model(X_batch)

            train_prediction = torch.cat((train_prediction,f_X))
            train_y = torch.cat((train_y,y_batch))

            optimizer.zero_grad()

            #loss = criterion(f_X, target=y_batch)
            loss = criterion(f_X, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        spearman = spearmanr(train_prediction.detach().cpu(),train_y.detach().cpu())
        print("train Spearman : ",spearman)
        print("Mean training loss: ",train_loss /len(train_dataset))

        time.sleep(2)

        print("validation...")
        model.eval()

        val_loss = 0
        val_prediction = torch.Tensor([]).to(device)
        val_y = torch.Tensor([]).to(device)

        with torch.no_grad():
            for X_batch, y_batch in val_loader:

                if torch.isnan(X_batch).any():
                    continue

                X_batch,y_batch = X_batch.to(device), y_batch.to(device)

                f_X = model(X_batch)

                val_prediction = torch.cat((val_prediction,f_X))
                val_y = torch.cat((val_y,y_batch))

                loss = criterion(f_X, y_batch)

                val_loss += loss.item()
            
            spearman = spearmanr(val_prediction.detach().cpu(),val_y.detach().cpu()).correlation
            print("validation Spearman : ",spearman)
            print("Mean validation loss: ",val_loss /len(val_dataset))

            if spearman > best_spearman and spearman > 0.74:
                best_spearman = spearman
                print("best spearman for now : ",spearman)
                torch.save(model.state_dict(),'/home/vegeta/Downloads/ML4G_Project_1_Data/my_checkpoints/actually_validated_weights.pth')

        time.sleep(2)
    

def main():
    train_val(train_loader, val_loader, model)

