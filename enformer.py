import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [0,1,2,3,4,5,6, ...]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) # [1, 0.86, 0.74, ... 0.0001]
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x: torch.Tensor):
        return x + self.encoding[:, :x.size(1)]   # only the first x.size(1) positions are actually used
        # broadcasting is used





class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_prob):
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
        self.conv_block2 = ConvBlock(dim_in, dim_out, kernel_size=1)
        self.pool = AttentionPool(dim_out)

    def forward(self,x):
        x = self.conv_block1(x)
        x1 = self.conv_block2(x)
        x = x + x1
        x = self.pool(x)
        return x


class Enformer(nn.Module):
    def __init__(self, conv_filters, dim, linear):
        super(Enformer, self).__init__()
        self.stem = Stem(dim // 2)
        conv_tower = []
        for dim_in, dim_out in zip(conv_filters[:-1], conv_filters[1:]):
            conv_tower.append(ConvLayer(dim_in, dim_out))
        self.conv_tower  = nn.Sequential(*conv_tower)

        self.transformer = Transformer(dim)

        self.linear1 = nn.Linear(dim, linear)
        self.linear2 = nn.Linear(linear,1)

    def forward(self, x ):
        x = self.stem(x)
        x = self.conv_tower(x)
        x = self.transformer(x)
        x = F.gelu( self.linear1(x) )
        x = self.linear2(x)
        return x



dim = 128

conv_n = 5
conv_filters = np.linspace(dim // 2, dim, num=5 ).astype(int)

linear_dim = 100


model = Enformer(conv_filters, dim, linear_dim)

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


batch_size = 4
num_epochs = 5
learning_rate = 0.0001


train_dataset = SimpleDataset(X_train, y_train)
val_dataset = SimpleDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)





