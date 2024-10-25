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


def retrieve_histone_data(bw_file, X,  histone_index, gene_names, chroms, tss_centers, strands, gex, halfspan ,downsample_size):  # can also be used for dnase

    for gene_index in tqdm(range(len(gene_names))):

        try:
            seq = np.array(bw_file.values(chroms[gene_index], tss_centers[gene_index] - halfspan, tss_centers[gene_index] + halfspan), dtype=np.float32)

        except:

            if tss_centers[gene_index] - halfspan < 0:
                seq = bw_file.values(chroms[gene_index], 0, tss_centers[gene_index] + halfspan)
                seq =  [0] *  (halfspan - tss_centers[gene_index] )   + seq 

            else :
                chrom_len = bw_file.chroms().get(chroms[gene_index])
                seq = bw_file.values(chroms[gene_index], tss_centers[gene_index] - halfspan , chrom_len) 
                seq = seq +  [0] * (halfspan - (chrom_len - tss_centers[gene_index]) ) 

        if downsample_size > 0:
            downsampled = F.interpolate(torch.tensor(seq).unsqueeze(0).unsqueeze(0), size=downsample_size, mode='linear', align_corners=False).squeeze().detach().numpy()
            X[gene_index, histone_index] = downsampled
        else:
            X[gene_index, histone_index] = seq


def retrieve_all_histones(base_dir, histones_list, cell_line, X, gene_names, chroms, tss_centers, strands, gex, halfspan=10000, downsample=-1):
    for histone_index, histone in enumerate(histones_list):

        bigwig_file_path = os.path.join(base_dir,histone+'-bigwig','X'+str(cell_line)+'.bigwig')

        if Path(bigwig_file_path).exists():

            bw = pyBigWig.open(bigwig_file_path)

        else:

            bigwig_file_path = os.path.join(base_dir,histone+'-bigwig','X'+str(cell_line)+'.bw')
            bw = pyBigWig.open(bigwig_file_path)

        retrieve_histone_data(bw, X,  histone_index, gene_names, chroms, tss_centers, strands, gex, halfspan, downsample)

        bw.close()

        print("histone "+histone + ' done')


def main():

    pass





