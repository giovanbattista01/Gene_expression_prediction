import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import time
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pyBigWig

from histone_data import retrieve_all_histones
from histone_data import retrieve_histone_data
from histone_data import retrieve_histone_data_around_gene

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


def generate_gene_x_y():
    gene_names, chroms, _, gene_coords,  strands, gex = load_gene_info(mode,cell_line)
    X = np.memmap('X_data.memmap', dtype='float32', mode='w+', shape=(len(gene_names), len(histones_list) + 1, chosen_dim))

    retrieve_all_histones(base_dir, histones_list, cell_line, X, gene_names, chroms, None, strands, gex,gene_coords=gene_coords, halfspan=2000,downsample_dim=20000)

    dnase_path = os.path.join(base_dir,'DNase-bigwig','X'+str(cell_line)+'.bw')
    dnase_file = pyBigWig.open(dnase_path)
    retrieve_histone_data_around_gene(dnase_file, X,  len(histones_list), gene_names, chroms, gene_coords, strands, gex)

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



def main():
    create_gene_data()






main()