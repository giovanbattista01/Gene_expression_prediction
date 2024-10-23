import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import time

import pandas as pd


"""class ConvNetModel(nn.Module):
    def __init__(self,width, nhistones, nfilters, filtsize, padding, poolsize, n_states_linear1, n_states_linear2, noutputs):
        super(ConvNetModel, self).__init__()
        self.conv1 = nn.Conv1d(nhistones, nfilters, filtsize, padding=padding)
        self.pool = nn.MaxPool1d(poolsize)
        self.fc1 = nn.Linear(int( width * nfilters // poolsize ), n_states_linear1)
        self.fc2 = nn.Linear(n_states_linear1, n_states_linear2)
        self.fc3 = nn.Linear(n_states_linear2, noutputs)
        self.dropout = nn.Dropout(0.4)

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

X_path = '/home/vegeta/Downloads/ML4G_Project_1_Data/my_solution/X3.npy'

X = torch.from_numpy(np.load(X_path)).float()


width = X.shape[-1]
nhistones = 5
nfilters = 50
filtsize = 10
poolsize = 5
padding = filtsize // 2
n_states_linear1 = 625  #625   #2000
n_states_linear2 = 125  #125   #1000
noutputs = 1

weights_path = '/home/vegeta/Downloads/ML4G_Project_1_Data/my_solution/final_weights.pth'
model = ConvNetModel(width,nhistones, nfilters, filtsize, padding, poolsize, n_states_linear1, n_states_linear2, noutputs)
model.load_state_dict(torch.load(weights_path))


batch_size = 16
learning_rate = 0.001


predictions = torch.zeros((X.shape[0]))


def predict(model, X, predictions):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        X = X.to(device)
        predictions = predictions.to(device)
        model = model.to(device)

        model.eval()

        for i in range(0,X.shape[0]//batch_size):

            X_batch = X[i*batch_size : (i+1)*batch_size]
            predictions[i*batch_size : (i+1)*batch_size] = model(X_batch)

        return predictions


pred = predict(model,X,predictions).detach().cpu().numpy()"""

### TODO: FIX NAN ISSUE!!! ###

pred  = np.load('/home/vegeta/Downloads/ML4G_Project_1_Data/preds.npy')


path_test = "/home/vegeta/Downloads/ML4G_Project_1_Data/CAGE-train/CAGE-train/X3_test_info.tsv" 
test_genes = pd.read_csv(path_test, sep='\t')
gene_names= test_genes['gene_name']

assert isinstance(pred, np.ndarray), 'Prediction array must be a numpy array'
assert np.issubdtype(pred.dtype, np.number), 'Prediction array must be numeric'
assert pred.shape[0] == len(test_genes), 'Each gene should have a unique predicted expression'

save_dir = '/home/vegeta/Downloads/ML4G_Project_1_Data/my_solution'  
file_name = 'gex_predicted.csv'         
zip_name = "Landolina_GiovanBattista_Project1.zip" 
save_path = f'{save_dir}/{zip_name}'
compression_options = dict(method="zip", archive_name=file_name)

test_genes['gex_predicted'] = pred.tolist()
test_genes[['gene_name', 'gex_predicted']].to_csv(save_path, compression=compression_options)


    

