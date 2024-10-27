import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import time
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pyBigWig
from pyfaidx import Fasta


from histone_data import retrieve_all_histones
from histone_data import retrieve_histone_data
from histone_data import retrieve_histone_data_around_gene

from dna_data import one_hot_encode

def load_gene_info(mode,cell_line):   # mode can be train,val or test, cell_line can be 1,2,3


    base_dir = "/home/vegeta/Downloads/ML4G_Project_1_Data/CAGE-train/CAGE-train/"
    file_string = 'X'+str(cell_line)+'_'+mode

    
    info_path = os.path.join(base_dir, file_string + '_info.tsv' )
    y_path = os.path.join(base_dir, file_string + '_y.tsv' )

    info = pd.read_csv(info_path, sep='\t')
    y = pd.read_csv(y_path, sep='\t')

    gene_names = info['gene_name']
    chrs = info['chr']
    tss_centers = [ int(( info['TSS_start'][i]+info['TSS_end'][i] ) // 2) for i in range(len(gene_names)) ]
    gene_coords = [(info['gene_start'][i], info['gene_end'][i]) for i in range(len(gene_names))]
    strands = info['strand']
    gex = y['gex']

    return gene_names, chrs, tss_centers, gene_coords , strands, gex


def generate_tss_x_y(histones_list, base_dir='/home/vegeta/Downloads/ML4G_Project_1_Data', mode='train', cell_line=1, halfspan=10000):   # creates dataset based on just tss centers positions
    gene_names, chroms, tss_centers, _ , strands, gex = load_gene_info(mode,cell_line)
    #X = np.zeros((len(gene_names), len(histones_list) + 1 , halfspan *2 ))  # number of histone marks + 1 for dnase
    X = np.memmap('X_data.memmap', dtype='float32', mode='w+', shape=(len(gene_names), len(histones_list) + 1, halfspan *2))


    retrieve_all_histones(base_dir, histones_list, cell_line, X, gene_names, chroms, tss_centers, strands, gex)

    dnase_path = os.path.join(base_dir,'DNase-bigwig','X'+str(cell_line)+'.bw')
    dnase_file = pyBigWig.open(dnase_path)
    retrieve_histone_data(dnase_file, X,  len(histones_list), gene_names, chroms, tss_centers, strands, gex)

    gex = np.array(gex)
    gex[np.isnan(gex)] = 0
    return X,gex

def create_tss_data():
    base_dir = '/home/vegeta/Downloads/ML4G_Project_1_Data/'
    save_dir = base_dir + 'tss_data/'

    modes = ['train','val']
    cell_lines = [1,2]

    histones_list = ['H3K4me3','H3K4me1','H3K36me3','H3K9me3','H3K27me3']

    for mode in modes:
        for cell_line in cell_lines:
            X,y = generate_tss_x_y(histones_list,mode=mode,cell_line=cell_line)
            np.save(save_dir + 'X' + str(cell_line) + '_' + mode, X)
            np.save(save_dir + 'y' + str(cell_line) + '_' + mode, y)


def generate_gene_x_y(histones_list, base_dir='/home/vegeta/Downloads/ML4G_Project_1_Data', mode='train', cell_line=1, halfspan=2000, chosen_dim=20000):
    gene_names, chroms, _, gene_coords,  strands, gex = load_gene_info(mode,cell_line)
    X = np.memmap('X_data.memmap', dtype='float32', mode='w+', shape=(len(gene_names), len(histones_list) + 1, chosen_dim))

    retrieve_all_histones(base_dir, histones_list, cell_line, X, gene_names, chroms, None, strands, gex,gene_coords=gene_coords, halfspan=halfspan,downsample_size=chosen_dim)

    dnase_path = os.path.join(base_dir,'DNase-bigwig','X'+str(cell_line)+'.bw')
    dnase_file = pyBigWig.open(dnase_path)
    retrieve_histone_data_around_gene(dnase_file, X,  len(histones_list), gene_names, chroms, gene_coords, strands, gex, halfspan=halfspan ,downsample_size=chosen_dim)

    gex = np.array(gex)
    gex[np.isnan(gex)] = 0
    return X,gex

def create_gene_data():
    base_dir = '/home/vegeta/Downloads/ML4G_Project_1_Data/'
    save_dir = base_dir + 'gene_data/'

    modes = ['train','val']
    cell_lines = [1,2]

    histones_list = ['H3K4me3','H3K4me1','H3K36me3','H3K9me3','H3K27me3']

    for mode in modes:
        for cell_line in cell_lines:
            X,y = generate_gene_x_y(histones_list,mode=mode,cell_line=cell_line)
            np.save(save_dir + 'X' + str(cell_line) + '_' + mode, X)
            np.save(save_dir + 'y' + str(cell_line) + '_' + mode, y)


# dataset with histones, dnase, positional encoding and dna information: 5+1+2+4=12
def create_augmented_dataset(mode,cell_line):
    seq_len = 5000
    stride = 3000
    max_iter = 10
    base_dir= '/home/vegeta/Downloads/ML4G_Project_1_Data/'
    save_dir = base_dir + 'augmented_data/'
    h5_file = h5py.File(save_dir+mode+str(cell_line)+'_augmented_dataset.h5', 'w')
    hg38_path = '/home/vegeta/Downloads/hg38.fa'

    genome = Fasta(hg38_path)

    gene_names, chroms, tss_centers, gene_coords , strands, gex = load_gene_info(mode,cell_line)

    dataset_X = h5_file.create_dataset(
        'X',            
        shape=(0, 12, seq_len ),  
        maxshape=(None, 12, seq_len),
        dtype='float32',       
        compression='gzip'
    )

    dataset_y = h5_file.create_dataset(
        'y',            
        shape=(0,),  
        maxshape=(None,),
        dtype='float32',       
        compression='gzip'
    )

    signals_list = ['H3K4me3','H3K4me1','H3K36me3','H3K9me3','H3K27me3','DNase']
        
    bw_file_list = []

    for signal in signals_list:
        bigwig_file_path = os.path.join(base_dir,signal+'-bigwig','X'+str(cell_line)+'.bigwig')

        if Path(bigwig_file_path).exists():

            bw = pyBigWig.open(bigwig_file_path)

        else:

            bigwig_file_path = os.path.join(base_dir,signal+'-bigwig','X'+str(cell_line)+'.bw')
            bw = pyBigWig.open(bigwig_file_path)

        bw_file_list.append(bw)

    for i in tqdm(range(len(gene_names))):
        
        c1, c2 = gene_coords[i][0], gene_coords[i][1]
        tss_center = tss_centers[i]
        gene_len = c2 - c1
        n_shards = (gene_len - seq_len) // stride + 1
        if n_shards < 1:
            continue

        l,r = c1, c1+seq_len

        for j in range(n_shards):
            if j >= max_iter:
                break

            dataset_X.resize(dataset_X.shape[0] + 1, axis=0)
            dataset_y.resize(dataset_y.shape[0] + 1, axis=0)
            
            for k,bw in enumerate(bw_file_list):

                seq = np.array(bw.values(chroms[i],l,r), dtype=np.float32)
                dataset_X[-1,k,:] = seq
            # positional encoding here (tss and gene coords)
            tss_encoding = np.arange(l,r) - tss_center
            gene_encoding = np.arange(l,r) - c1

            dataset_X[-1,len(bw_file_list),:] =  tss_encoding
            dataset_X[-1,len(bw_file_list)+1,:] =  gene_encoding

            # finally, the dna sequence:

            dna_seq = genome[chroms[i]][l:r].seq
            ohe = one_hot_encode(dna_seq)

            for base_index in range(4):
                dataset_X[-1,-base_index-1,:] = ohe[:,base_index]

            l,r = l+stride, r+stride

            dataset_y[-1] = gex[i]

    h5_file.close()





def main():

    base_dir = '/home/vegeta/Downloads/ML4G_Project_1_Data/_data/'


    create_augmented_dataset('train',1)


main()