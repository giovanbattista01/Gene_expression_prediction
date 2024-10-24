import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
import h5py
from tqdm import tqdm
import torch.optim as optim


torch.set_printoptions(sci_mode=False)

embed_dim = 128  
num_heads = 8    
seq_length = 10  
batch_size = 4  
dropout_prob = 0.5


# 128-bp resolution recommended



class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super(AttentionPool, self).__init__()
        self.pool_size = pool_size

        self.to_attn_logits = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        nn.init.dirac_(self.to_attn_logits.weight)  # Initialize weights to perform identity mapping
        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)  # Scale weights for attention

    def forward(self, x):
        b, d, n = x.shape  # Assume x is of shape (batch_size, dim, seq_len)

        """remainder = n % self.pool_size
        needs_padding = remainder > 0

        # Padding if the sequence length is not divisible by pool_size
        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
        """

        # Reshape for pooling
        x = x.view(b, d, n // self.pool_size, self.pool_size)
        
        # Apply attention logits using 2D conv
        logits = self.to_attn_logits(x)  # Apply 1x1 convolution to get attention logits
        
        # Apply softmax to get attention weights
        attn = logits.softmax(dim=-1)

        # Return weighted sum of the pooled output
        return (x * attn).sum(dim=-1)  # Sum across the pooling dimension




class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to('cuda')
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [0,1,2,3,4,5,6, ...]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) # [1, 0.86, 0.74, ... 0.0001]
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x: torch.Tensor):
        return x + self.encoding[:,:x.size(1),:]   # only the first x.size(2) positions are actually used
        # broadcasting is used





class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout_prob=0.2):
        super(Transformer, self).__init__()
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 2 )
        self.linear2 = nn.Linear(embed_dim * 2, embed_dim )

    def forward(self, x):
        x1 = self.pos_encoding(x)
        x1 = self.layer_norm(x1)
        x1, _ = self.attention(x1, x1, x1)
        x1 = self.dropout(x1)
        x = x + x1  # Residual connection

        x2 = self.layer_norm(x)
        x2 = self.linear1(x2)
        x2 = self.dropout(x2)
        x2 = F.relu(x2)
        x2 = self.linear2(x2)
        x2 = self.dropout(x2)

        return x + x2  # Residual connection


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim_in),
            nn.GELU(),
            nn.Conv1d(dim_in, dim_out, kernel_size, padding=kernel_size // 2)
        )

    def forward(self, x):
        return self.block(x)

class Stem(nn.Module):
    def __init__(self,half_dim):
        super(Stem, self).__init__()
        self.first_conv = nn.Conv1d(4,half_dim,15,padding=7)
        self.conv_block = ConvBlock(half_dim,half_dim)
        self.pool = AttentionPool(half_dim)

    def forward(self, x):
        x = self.first_conv(x)
        x1 = self.conv_block(x)
        x = x + x1                 # residual connection 
        return self.pool(x)


class ConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConvLayer,self).__init__()
        self.conv_block1 = ConvBlock(dim_in, dim_out, kernel_size=5)
        self.conv_block2 = ConvBlock(dim_out, dim_out, kernel_size=1)
        self.pool = AttentionPool(dim_out)

    def forward(self,x):
        x = self.conv_block1(x)
        x1 = self.conv_block2(x)
        x = x + x1
        x = self.pool(x)
        return x


class Enformer(nn.Module):
    def __init__(self, initial_seq_len, conv_filters, dim, linear):
        super(Enformer, self).__init__()
        self.stem = Stem(dim // 2)
        conv_tower = []
        for dim_in, dim_out in zip(conv_filters[:-1], conv_filters[1:]):
            conv_tower.append(ConvLayer(dim_in, dim_out))
        self.conv_tower  = nn.Sequential(*conv_tower)

        self.transformer = Transformer(dim)

        self.linear1 = nn.Linear(dim * (initial_seq_len // pow(2,len(conv_filters))) , linear)
        self.linear2 = nn.Linear(linear,1)

    def forward(self, x ):
        x = x.transpose(1,2)
        x = self.stem(x)
        x = self.conv_tower(x)
        x = x.transpose(1,2)
        x = self.transformer(x)
        x = x.reshape(x.size(0),x.size(1)*x.size(2))
        x = F.gelu( self.linear1(x) )
        x = self.linear2(x)
        return x.squeeze()



class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, transform=None):
        
        self.h5_file_path = h5_file_path
        self.transform = transform
        
        self.h5_file = h5py.File(h5_file_path, 'r')

        self.X = self.h5_file['dna_data']
        self.y = self.h5_file['gex_data']

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        if self.transform:
            data = self.transform(data)
        
        return X,y
    
    def close(self):
        self.h5_file.close()


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



batch_size = 4
num_epochs = 5
learning_rate = 0.0001

dim = 128

conv_n = 7
conv_filters = np.linspace(dim // 2, dim, num=conv_n).astype(int)

linear_dim = 100
seq_len = 200000


model = Enformer(seq_len,conv_filters, dim, linear_dim)

base_dir = '/home/vegeta/Downloads/ML4G_Project_1_Data/my_dna_data/'

train_path = base_dir + 'data1.h5'
val_path = base_dir + 'data1_val.h5'

train_dataset = HDF5Dataset(train_path)
val_dataset = HDF5Dataset(val_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def train_val (train_loader, val_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    #criterion = nn.MSELoss(reduction='mean')
    criterion = RankNet1DLoss(model=model, l2_lambda=0.00)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_spearman = 0

    for epoch in range(num_epochs):
        print("epoch : "+str(epoch))

        print("training...")
        model.train()

        train_loss = 0
        train_prediction = torch.Tensor([]).to(device)
        train_y = torch.Tensor([]).to(device)

        for X_batch, y_batch in tqdm(train_loader):

            """if torch.isnan(X_batch).any():
                continue"""

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
            print("Mean validation loss: ",val_loss /len(val_dataset))

            if spearman > best_spearman and spearman > 0.74:
                best_spearman = spearman
                print("best spearman for now : ",spearman)
                torch.save(model.state_dict(),'/home/vegeta/Downloads/ML4G_Project_1_Data/my_checkpoints/actually_validated_weights.pth')

        time.sleep(2)

train_val(train_loader, val_loader, model)






