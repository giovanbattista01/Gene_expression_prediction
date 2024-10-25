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
    strands = info['strand']
    gex = y['gex']

    return gene_names, chrs, tss_centers, strands, gex

def get_dnase():



def create_tss_dataset(histones_list, base_dir='/home/vegeta/Downloads/ML4G_Project_1_Data', mode='train', cell_line=1, halfspan=10000):   # creates dataset based on just tss centers positions
    gene_names, chroms, tss_centers, strands, gex = load_gene_info(mode,cell_line)
    X = np.zeros((len(gene_names), len(histones_list) + 1 , halfspan  ))  # number of histone marks + 1 for dnase

    retrieve_all_histones(base_dir, histones_list, X, gene_names, chroms, tss_centers, strands, gex)

    dnase_path = os.path.join(base_dir,'DNase-bigwig','X'+str(cell_line)+'.bw')
    dnase_file = pyBigWig.open(dnase_path)
    retrieve_histone_data(dnase_file, X,  len(histones_list), gene_names, chroms, tss_centers, strands, gex)

    


def main():
    pass




main()