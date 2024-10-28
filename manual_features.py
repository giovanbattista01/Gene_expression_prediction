import pandas as pd
import numpy as np
from data import load_gene_info
from histone_data import retrieve_histone_data
import os
from pathlib import Path
import pyBigWig
from tqdm import tqdm
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from scipy.stats import spearmanr

from Bio import SeqIO


def calculate_gc_content(sequence):
    """Calculate the GC content of a given DNA sequence."""
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence) if len(sequence) > 0 else 0

def calculate_cpg_ratio(sequence):
    """Calculate the observed-to-expected CpG ratio for a given sequence."""
    c_count = sequence.count("C")
    g_count = sequence.count("G")
    cg_count = sequence.count("CG")
    if c_count * g_count > 0:
        return (cg_count * len(sequence)) / (c_count * g_count)
    else:
        return 0

def detect_cpg_islands(sequence, window_size=200, gc_threshold=0.5, cpg_ratio_threshold=0.6):
    """Detect CpG islands in a given DNA sequence using a sliding window approach."""
    cpg_islands = []
    for i in range(len(sequence) - window_size + 1):
        window_seq = sequence[i:i + window_size]
        gc_content = calculate_gc_content(window_seq)
        cpg_ratio = calculate_cpg_ratio(window_seq)
        
        if gc_content >= gc_threshold and cpg_ratio >= cpg_ratio_threshold:
            cpg_islands.append((i, i + window_size))
    return cpg_islands



def get_feature_names(signals_list, thresholds, spans):
    list_gene_features = ['gene_len', 'strand', 'closeness_to_other_genes']

                                            ### LEGEND ###

    # tss_Peak_-50:50_1  means this is a binary feature that it's 1 if max(signal) around the tss (-50:50) > 1

    list_tss_features = [
        f"{signal}_tss_Peak_-{s}:{s}_{t}" 
        for signal in signals_list 
        for t in thresholds 
        for s in spans
    ]

    # additionally, let's keep tracks of peaks in the gene area

    for signal in signals_list:
        for t in thresholds:
            list_gene_features.append(f"{signal}_gene_Peak_{t}")
        list_gene_features.extend([f"{signal}_max", f"{signal}_min", f"{signal}_max_loc", f"{signal}_min_loc"])


    # additional features: 

    additional_features = []

    for i,signal in enumerate(signals_list):

        additional_features.extend([f"{signal}_Peak-tss_distance",f"{signal}_promoter_Peak",f"{signal}_MaxGradient",f"{signal}_AvgGradient" ])

        for j,signal2 in enumerate(signals_list):
            if i >= j:
                continue
            additional_features.append(f"{signal}_{signal2}_Ratio")

        

    list_features = list_gene_features + list_tss_features + additional_features

    return list_features


def get_features_from_seq(dataset, gene_index, bw_files, chroms, tss_centers, gene_coords , strands,signals_list, thresholds, spans):

    c =  gene_coords[gene_index]
    strand = int((strands[gene_index]=='+')) 
    gene_len = c[1] - c[0]

    dataset['gene_len'][gene_index] = gene_len
    dataset['strand'][gene_index] = strand

    lefts, rights = np.array([coord[0] for coord in gene_coords]), np.array([coord[1] for coord in gene_coords])
    left_distances, right_distances = c[0] - rights, lefts - c[1]

    try:
        min_distance = min(np.min(left_distances[left_distances > 0]), np.min(right_distances[right_distances > 0]))
    except ValueError:
        min_distance = 1e10
    dataset['closeness_to_other_genes'][gene_index] = min_distance


    for i,bw_file in enumerate(bw_files):

        signal = signals_list[i]
        for t in thresholds:
            # tss features
            for s in spans:
                seq_tss = np.array(bw_file.values(chroms[gene_index],tss_centers[gene_index]-s,tss_centers[gene_index]+s), dtype=np.float32)
                peak = ( np.max(seq_tss) > t ).astype(int)
                dataset[f"{signal}_tss_Peak_-{s}:{s}_{t}"][gene_index] = peak

            
            # gene features

            seq_gene = np.array(bw_file.values(chroms[gene_index], c[0], c[1]), dtype=np.float32)
            dataset[f"{signal}_gene_Peak_{t}"][gene_index] = int(np.max(seq_gene) > t)
            dataset[f"{signal}_max"][gene_index] = np.max(seq_gene)
            dataset[f"{signal}_min"][gene_index] = np.min(seq_gene)
            dataset[f"{signal}_max_loc"][gene_index] = np.argmax(seq_gene) / gene_len
            dataset[f"{signal}_min_loc"][gene_index] = np.argmin(seq_gene) / gene_len

            # distance of the peak from the tss

            dataset[f"{signal}_Peak-tss_distance"][gene_index] = (np.argmax(seq_gene) + c[0]) - (tss_centers[gene_index] - 25)


    # ADDITIONAL FEATURES

    promoter_peaks = []

    # promoter region peaks

    for i,file in enumerate(bw_files):
        tss = tss_centers[gene_index]-25
        promoter_size = 1000
        promoter_region  = np.array(file.values(chroms[gene_index],tss - promoter_size,tss), dtype=np.float32)
        peak = np.max(promoter_region)
        signal = signals_list[i]
        dataset[f"{signal}_promoter_Peak"][gene_index] = peak

        promoter_peaks.append(peak) 

    # histone modification ratios

    for i,file1 in enumerate(bw_files):
        for j,file2 in enumerate(bw_files):
            if i >= j:
                continue
            signal1 = signals_list[i]
            signal2 = signals_list[j]

            dataset[f"{signal1}_{signal2}_Ratio"][gene_index] = promoter_peaks[i] / promoter_peaks[j]

    # max gradient around the tss can help understanding the region structure

    for i,file in enumerate(bw_files):
        tss = tss_centers[gene_index]- 25
        s = 1000
        seq_tss = np.array(bw_file.values(chroms[gene_index],tss-s,tss+s), dtype=np.float32) 

        tss_gradient = np.gradient(seq_tss)
        max_gradient = np.max(tss_gradient)
        average_gradient = np.mean(tss_gradient)

        signal = signals_list[i]

        dataset[f"{signal}_MaxGradient"][gene_index] = max_gradient
        dataset[f"{signal}_AvgGradient"][gene_index] = average_gradient


def build_dataset(mode,cell_line):

    print("building manual features dataset")

    base_dir = "/home/vegeta/Downloads/ML4G_Project_1_Data/"
    save_dir = "/home/vegeta/Downloads/ML4G_Project_1_Data/manual_feature_data/"


    signals_list = ['H3K4me3','H3K4me1','H3K36me3','H3K9me3','H3K27me3','DNase']

    thresholds = [1,5,10]

    spans = [50,500,5000]

    gene_names, chroms, tss_centers, gene_coords , strands, gex = load_gene_info(mode,cell_line)

    feature_names = get_feature_names(signals_list, thresholds, spans)

    with h5py.File(save_dir + f'X{cell_line}_{mode}.h5', 'w') as hdf5_file:
        dataset = {name: hdf5_file.create_dataset(name, shape=(len(gene_names),), dtype='f') for name in feature_names}

        # Load bigWig files
        print("loading bigwig files")

        bw_files = []
        for signal in signals_list:
            bigwig_file_path = os.path.join(base_dir, f"{signal}-bigwig", f"X{cell_line}.bigwig")
            if not Path(bigwig_file_path).exists():
                bigwig_file_path = os.path.join(base_dir, f"{signal}-bigwig", f"X{cell_line}.bw")
            bw_files.append(pyBigWig.open(bigwig_file_path))

        # Fill dataset
        print("filling dataset")
        for gene_index in tqdm(range(len(gene_names))):
            get_features_from_seq(dataset, gene_index, bw_files, chroms, tss_centers, gene_coords, strands, signals_list, thresholds, spans)

        # Close all bigWig files
        for bw in bw_files:
            bw.close()


def get_data(feature_names, train, val, n_train, n_val,skip=-1):
    if skip != -1:
        new  = []
        for f in feature_names:
            if 'H3' in f:
                new.append(f)
        feature_names = new
        print(feature_names)
    
    X_train = np.zeros((n_train,len(feature_names)))
    X_val = np.zeros((n_val,len(feature_names)))

    for i,feature in enumerate(feature_names):
        X_train[:,i] = train[feature][:]
        X_val[:,i] = val[feature][:]

    X_train[np.isnan(X_train)] = 0
    X_val[np.isnan(X_val)] = 0

    X_train[X_train==np.inf] = 1e10
    X_val[X_val==np.inf] = 1e10

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val


def train_val():
    base_dir = "/home/vegeta/Downloads/ML4G_Project_1_Data/manual_feature_data/"
    train = h5py.File(base_dir+"X2_train.h5",'r')
    val = h5py.File(base_dir+"X2_val.h5",'r')

    y_train = np.load("/home/vegeta/Downloads/ML4G_Project_1_Data/my_histone_data/old_data/y2_train.npy")
    y_val = np.load("/home/vegeta/Downloads/ML4G_Project_1_Data/my_histone_data/old_data/y2_val.npy")

    y_train[np.isnan(y_train)] = 0
    y_val[np.isnan(y_val)] = 0

    signals_list = ['H3K4me3','H3K4me1','H3K36me3','H3K9me3','H3K27me3','DNase']

    thresholds = [1,5,10]

    spans = [50,500,5000]


    feature_names = get_feature_names(signals_list, thresholds, spans)

    X_train, X_val = get_data(feature_names, train, val, y_train.shape[0],y_val.shape[0])

    y_train_bin = (y_train > 0 ).astype(int)
    y_val_bin = (y_val > 0 ).astype(int)


    log_reg = LogisticRegression(penalty='l2', C=0.01,max_iter=100000)  
    log_reg.fit(X_train, y_train_bin)


    y_pred = log_reg.predict(X_val)

    print(f"Accuracy: {accuracy_score(y_val_bin, y_pred)}")
    print(classification_report(y_val_bin, y_pred))
    print("spearman bin ",spearmanr(y_val_bin,y_pred))
    print("spearman ",spearmanr(y_val,y_pred))


    n_important = 20

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    feature_importance_log_reg = np.abs(log_reg.coef_[0]) 
    importance_df_log_reg = pd.DataFrame({
        "Feature": X_train_df.columns,
        "Importance": feature_importance_log_reg
    }).sort_values(by="Importance", ascending=False)
    print("Feature Importance (Logistic Regression):\n", importance_df_log_reg[:n_important])

    top_features = importance_df_log_reg[:n_important]['Feature'].tolist()

    coefficients = log_reg.coef_[0]  # Coefficients for each feature
    for i, coef in enumerate(coefficients):
        print(f"Feature {i + 1}: {'Positive' if coef > 0 else 'Negative' if coef < 0 else 'Neutral'} effect (Coefficient = {coef})")

    
    print("retrain  with only important parameters ...")


    X_train, X_val = get_data(top_features, train, val, y_train.shape[0],y_val.shape[0])

    log_reg = LogisticRegression(penalty='l2', C=1.0,max_iter=100000)  
    log_reg.fit(X_train, y_train_bin)


    y_pred = log_reg.predict(X_val)

    print(f"Accuracy: {accuracy_score(y_val_bin, y_pred)}")
    print(classification_report(y_val_bin, y_pred))
    print("spearman bin ",spearmanr(y_val_bin,y_pred))
    print("spearman ",spearmanr(y_val,y_pred))


    """rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)  
    rf_classifier.fit(X_train, y_train_bin)

    y_pred = rf_classifier.predict(X_val)

    print(f"Accuracy: {accuracy_score(y_val_bin, y_pred)}")
    print(classification_report(y_val_bin, y_pred))
    print("Spearman correlation (binary): ", spearmanr(y_val_bin, y_pred))
    print("Spearman correlation (regression): ", spearmanr(y_val, y_pred))"""


    train.close()
    val.close()

def main():
    train_val()




main()