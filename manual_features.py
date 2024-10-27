import pandas as pd
import numpy as np
from data import load_gene_info
from histone_data import retrieve_histone_data
import os
from pathlib import Path
import pyBigWig
from tqdm import tqdm
import h5py


def get_features_names(signals_list, thresholds, spans):
    list_gene_features = ['gene_len', 'strand', 'closeness_to_other_genes']

    list_tss_features = [] 

                                            ### LEGEND ###

    # tss_Peak_-50:50_1  means this is a binary feature that it's 1 if max(signal) around the tss (-50:50) > 1

    for signal in signals_list:
        for t in thresholds:
            for s in spans:
                list_tss_features.append(f"{signal}_tss_Peak_-{s}:{s}_{t}")
        


    # additionally, let's keep tracks of peaks in the gene area

    for signal in signals_list:
        for t in thresholds:
            list_gene_features.append(f"{signal}_gene_Peak_{t}")
        list_gene_features.extend([f"{signal}_max", f"{signal}_min", f"{signal}_max_loc", f"{signal}_min_loc"])


    list_features = list_gene_features + list_tss_features 

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
        min_distance = np.inf
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
            dataset[f"{signal}_max_loc"][gene_index] = np.argmax(seq_gene)
            dataset[f"{signal}_min_loc"][gene_index] = np.argmin(seq_gene)

def build_dataset(mode,cell_line):

    print("building manual features dataset")

    base_dir = "/home/vegeta/Downloads/ML4G_Project_1_Data/"
    save_dir = "/home/vegeta/Downloads/ML4G_Project_1_Data/manual_feature_data/"


    signals_list = ['H3K4me3','H3K4me1','H3K36me3','H3K9me3','H3K27me3','DNase']

    thresholds = [1,5,10]

    spans = [50,500,5000]

    gene_names, chroms, tss_centers, gene_coords , strands, gex = load_gene_info(mode,cell_line)

    feature_names = get_features_names(signals_list, thresholds, spans)

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
def main():
    build_dataset('val',1)



main()