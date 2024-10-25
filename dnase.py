import numpy as np
import pyBigWig
import pandas as pd
import matplotlib.pyplot as plt
from histone_data import load_gene_info
from scipy.stats import spearmanr
import time
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score





def retrieve_dnase(chrs, lefts, rights, dnase_path='/home/vegeta/Downloads/ML4G_Project_1_Data/DNase-bigwig/X1.bw'):
    with pyBigWig.open(dnase_path) as dnase:
        dnase_values = []  # Use a list to gather results
        for i in tqdm(range(len(chrs))):
            try:
                # Get DNase values for the current region
                values = dnase.values(chrs[i], lefts[i], rights[i])
                dnase_values.append(values)
            except Exception as e:  # Catch specific exceptions
                print(f"Error retrieving values for {chrs[i]}:{lefts[i]}-{rights[i]}: {e}")

    return np.array(dnase_values)


def train_val(X_train,y_train,X_val,y_val,model,gex=None):
    if model == 'log_reg':
        log_reg_model = LogisticRegression(max_iter=2000)
        log_reg_model.fit(X_train, y_train)

        val_predictions = log_reg_model.predict(X_val)
        val_probabilities = log_reg_model.predict_proba(X_val)[:, 1]

        print("Accuracy:", accuracy_score(y_val, val_predictions))
        print("Spearman Correlation:",spearmanr(val_predictions, y_val))
        print("Spearman Correlation:",spearmanr(val_predictions, gex))
        return

    if model == 'rf':
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)  
        rf_model.fit(X_train, y_train)

        val_predictions = rf_model.predict(X_val)

        print("Accuracy:", accuracy_score(y_val, val_predictions))
        print("Spearman Correlation:", spearmanr(val_predictions, y_val))
        print("Spearman Correlation:", spearmanr(val_predictions, gex))
        return





def main():
    halfspan =  50 # 50 is good for log_reg, 5000 or more is good for rf

    gene_names, chrs, tss_centers, strands, gex = load_gene_info('train',1)

    scaler = StandardScaler()

    X_train = retrieve_dnase(chrs, [c - halfspan for c in tss_centers], [c + halfspan for c in tss_centers])

    y_train = (gex > 0).astype(int).to_numpy().ravel()
    #X_train = scaler.fit_transform(X_train)


    gene_names, chrs, tss_centers, strands, gex = load_gene_info('val',1)

    X_val = retrieve_dnase(chrs, [c - halfspan for c in tss_centers], [c + halfspan for c in tss_centers])

    y_val = (gex > 0).astype(int).to_numpy().ravel()
    #X_val = scaler.transform(X_val)

    model = 'log_reg'
    train_val(X_train,y_train,X_val,y_val,model,gex.to_numpy())




main()