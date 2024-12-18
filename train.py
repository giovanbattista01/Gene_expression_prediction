import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np
import time
import h5py
from scipy.stats import spearmanr

from cnns import ConvNetModel
from cnns import DeepConvNet
from enformer import Enformer

from sklearn.metrics import roc_auc_score, confusion_matrix


class BCE_L2(nn.Module):
    def __init__(self, model = None, l2_lambda=0):
        super(BCE_L2, self).__init__()
        self.l2_lambda = l2_lambda
        self.model = model

    def forward(self, y_pred, y_true):

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_pred, y_true)
        if self.model != None and self.l2_lambda > 0:

            l2_reg = 0.0
            for param in self.model.parameters():
                l2_reg += torch.norm(param, 2)

            loss = loss + self.l2_lambda * l2_reg

        return loss


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

class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, mode='train', transform=None):
        
        self.h5_file_path = h5_file_path
        self.transform = transform
        
        self.h5_file = h5py.File(h5_file_path, 'r')

        try:
            self.X = self.h5_file['X']
            self.y = self.h5_file['y']

        except:
            self.X = self.h5_file['X_'+mode]
            self.y = self.h5_file['y_'+mode]

        self.seq_len = self.X.shape[-1]
        self.num_samples = self.y.shape[0]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        
        X = torch.tensor(X, dtype=torch.float32)
        
        if self.transform == 'binary':
            y = 1 if y > 0 else 0
        y = torch.tensor(y, dtype=torch.float32)
        
        return X,y


def train_val (train_loader, val_loader, model, num_training_samples, num_val_samples, crit="rank_loss"):

    learning_rate = 0.0001
    num_epochs = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    if crit == "rank_loss":
        criterion = RankNet1DLoss(model=model, l2_lambda=0.05)
    elif crit  == "bce_loss":
        criterion = BCE_L2(model=model, l2_lambda=0.1)
    else:
        print("error")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_spearman = 0

    goal = 0.8
    save_dir = '/home/vegeta/Downloads/ML4G_Project_1_Data/my_checkpoints/best_weights.pth'

    for epoch in range(num_epochs):
        print("epoch : "+str(epoch))

        print("training...")
        model.train()

        train_loss = 0
        train_prediction = torch.Tensor([]).to(device)
        train_y = torch.Tensor([]).to(device)

        for X_batch, y_batch in tqdm(train_loader):

            if torch.isnan(X_batch).any():
                continue

            X_batch,y_batch = X_batch.to(device), y_batch.to(device)

            f_X = model(X_batch)

            train_prediction = torch.cat((train_prediction,f_X))
            train_y = torch.cat((train_y,y_batch))

            optimizer.zero_grad()

            loss = criterion(f_X, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            

        spearman = spearmanr(train_prediction.detach().cpu(),train_y.detach().cpu())
        print("train Spearman : ",spearman)
        print("Mean training loss: ",train_loss /num_training_samples)

        if crit == "bce_loss":
                train_y = train_y.detach().cpu()
                train_pred = torch.sigmoid(train_prediction).detach().cpu()
                train_pred = (train_pred  > 0.5).int()   
                tn, fp, fn, tp = confusion_matrix(train_y,train_pred).ravel()
                print("tn, fp, fn, tp : ",tn,fp,fn,tp)

        time.sleep(2)

        print("validation...")
        model.eval()

        val_loss = 0
        val_prediction = torch.Tensor([]).to(device)
        val_y = torch.Tensor([]).to(device)

        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader):

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
            print("Mean validation loss: ",val_loss /num_val_samples)

            if crit == "bce_loss":
                val_y = val_y.detach().cpu()
                val_pred = torch.sigmoid(val_prediction).detach().cpu()
                val_pred = (val_pred  > 0.5).int()   
                tn, fp, fn, tp = confusion_matrix(val_y, val_pred).ravel()
                print("tn, fp, fn, tp : ",tn,fp,fn,tp)

            #print(val_prediction[:10])

            if spearman > best_spearman and spearman > goal:
                best_spearman = spearman
                print("best spearman for now : ",spearman)
                torch.save(model.state_dict(),save_dir)

        time.sleep(2)

def ConvNetSetup():
    
    crit = "rank_loss"

    base_dir = '/home/vegeta/Downloads/ML4G_Project_1_Data/tss_data/'
    
    train_path =  base_dir + 'train_data1.h5'
    val_path = base_dir + 'val_data2.h5'

    #y_train = (y_train > 0).float()
    train_dataset = HDF5Dataset(train_path)

    #y_val = (y_val > 0).float()
    val_dataset = HDF5Dataset(val_path, mode='val')

    width = train_dataset.seq_len
    nhistones = 6
    nfilters = 50 #200 
    filtsize = 10 #20
    poolsize = 5
    padding = filtsize // 2
    n_states_linear1 = 100  #100  625 #2000
    n_states_linear2 = 20  #20  125  #1000
    noutputs = 1

    batch_size = 8

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=16)

    model = ConvNetModel(width, nhistones, nfilters, filtsize, padding, poolsize, n_states_linear1, n_states_linear2, noutputs)

    n_training  = train_dataset.num_samples
    n_validation  = val_dataset.num_samples

    return train_loader, val_loader, model, n_training, n_validation, crit


def enformerSetup():

    base_dir = '/home/vegeta/Downloads/ML4G_Project_1_Data/my_dna_data/'

    train_path = base_dir + 'data1.h5'
    val_path = base_dir + 'data1_val.h5'

    train_dataset = HDF5Dataset(train_path)
    val_dataset = HDF5Dataset(val_path)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    
    dim = 128
    conv_n = 8
    conv_filters = np.linspace(dim // 2, dim, num=conv_n).astype(int)
    linear_dim = 100
    seq_len = 200000


    model = Enformer(seq_len,conv_filters, dim, linear_dim)

    return train_loader, val_loader, model, len(train_dataset), len(val_dataset)


def DeepConvTssSetup():
    
    crit = "rank_loss"

    base_dir = '/home/vegeta/Downloads/ML4G_Project_1_Data/tss_data/'
    
    train_path =  base_dir + 'train_data1.h5'
    val_path = base_dir + 'val_data2.h5'

    #y_train = (y_train > 0).float()
    train_dataset = HDF5Dataset(train_path)

    #y_val = (y_val > 0).float()
    val_dataset = HDF5Dataset(val_path, mode='val')

    width = train_dataset.seq_len
    nfeatures = 12 #6
    filter_list = [5,50] 
    filtsize_list = [10,10]
    poolsize_list = [10,5]
    padding_list = [filtsize // 2 for filtsize in filtsize_list]
    n_states_linear1 = 50 
    n_states_linear2 = 20  
    noutputs = 1

    batch_size = 8

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=16)

    model = DeepConvNet(width, nfeatures, filter_list, filtsize_list, padding_list, poolsize_list, n_states_linear1, n_states_linear2, noutputs)

    n_training  = train_dataset.num_samples
    n_validation  = val_dataset.num_samples

    return train_loader, val_loader, model, n_training, n_validation, crit

def DeepConvAugmentedSetup():

    crit = "bce_loss"
    
    base_dir = '/home/vegeta/Downloads/ML4G_Project_1_Data/augmented_data/'
    train_path = base_dir + 'train1_augmented_dataset.h5'
    val_path = base_dir + 'val1_augmented_dataset.h5'
    

    #y_train = (y_train > 0).float()
    train_dataset = HDF5Dataset(train_path,transform='binary')

    #y_val = (y_val > 0).float()
    val_dataset = HDF5Dataset(val_path, mode='val',transform='binary')

    width = train_dataset.seq_len
    nfeatures = 12 #6
    filter_list = [50,100] 
    filtsize_list = [10,10]
    poolsize_list = [10,5]
    padding_list = [filtsize // 2 for filtsize in filtsize_list]
    n_states_linear1 = 600 
    n_states_linear2 = 100  
    noutputs = 1

    batch_size = 8

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=16)

    model = DeepConvNet(width, nfeatures, filter_list, filtsize_list, padding_list, poolsize_list, n_states_linear1, n_states_linear2, noutputs)

    n_training  = train_dataset.num_samples
    n_validation  = val_dataset.num_samples

    return train_loader, val_loader, model, n_training, n_validation, crit





def main():
    chosen_model = 'deep_conv_augmented'

    if chosen_model == 'conv_net':
        train_loader, val_loader, model, num_training_samples, num_val_samples, crit = ConvNetSetup()
    elif chosen_model == 'deep_conv_tss':
        train_loader, val_loader, model, num_training_samples, num_val_samples, crit = DeepConvTssSetup()
    elif chosen_model == 'deep_conv_augmented':
        train_loader, val_loader, model, num_training_samples, num_val_samples, crit = DeepConvAugmentedSetup()
    elif chosen_model == 'enformer':
        train_loader, val_loader, model, num_training_samples, num_val_samples, crit = enformerSetup()
    else:
        print("model is not valid")
        exit()
    
    train_val(train_loader, val_loader, model, num_training_samples, num_val_samples, crit = crit)


main()