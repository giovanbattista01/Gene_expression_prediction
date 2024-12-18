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

import pyBigWig


###   RETRIEVING INFO ON THE GENE EXPRESSION DATASET ###

def load_gene_info(mode,chrom_set):   # mode can be train,val or test, chrom_set can be 1,2,3

    base_path = "/home/vegeta/Downloads/ML4G_Project_1_Data/CAGE-train/CAGE-train/"
    file_string = 'X'+str(chrom_set)+'_'+mode
    
    info_path = os.path.join(base_path, file_string + '_info.tsv' )

    info = pd.read_csv(info_path, sep='\t')

    gene_names = info['gene_name']
    chrs = info['chr']
    tss_centers = [ int(( info['TSS_start'][i]+info['TSS_end'][i] ) // 2) for i in range(len(gene_names)) ]
    strands = info['strand']

    return gene_names, chrs, tss_centers, strands


gene_names, chroms, tss_centers, strands = load_gene_info('test',3)


### RETRIEVING HISTONE DATA ###



histones_used = ['H3K4me3','H3K4me1','H3K36me3','H3K9me3','H3K27me3']

base_path = '/home/vegeta/Downloads/ML4G_Project_1_Data'

halfspan = 100000 #kb
final_size = 2000

X = np.zeros((len(gene_names), len(histones_used), final_size  ))


def retrieve_histones_data(bw_file, histone_index):

    halfspan  = 100000

    for gene_index in range(len(gene_names)):

        print(histone_index, gene_index)

        try:
            seq = bw_file.values(chroms[gene_index], tss_centers[gene_index] - halfspan, tss_centers[gene_index] + halfspan )

        except:

            if tss_centers[gene_index] - halfspan < 0:
                seq = bw_file.values(chroms[gene_index], 0, tss_centers[gene_index] + halfspan)
                seq =  [0] *  (halfspan - tss_centers[gene_index] )   + seq 

            else :
                chrom_len = bw_file.chroms().get(chroms[gene_index])
                seq = bw_file.values(chroms[gene_index], tss_centers[gene_index] - halfspan , chrom_len) 
                seq = seq +  [0] * (halfspan - (chrom_len - tss_centers[gene_index]) ) 


        downsampled = F.interpolate(torch.tensor(seq).unsqueeze(0).unsqueeze(0), size=final_size, mode='linear', align_corners=False).squeeze().detach().numpy()
        X[gene_index,histone_index] = downsampled


chrom_set = 3

for histone_index, histone in enumerate(histones_used):

    bigwig_file_path = os.path.join(base_path,histone+'-bigwig','X'+str(chrom_set)+'.bigwig')

    if Path(bigwig_file_path).exists():

        bw = pyBigWig.open(bigwig_file_path)

    else:

        bigwig_file_path = os.path.join(base_path,histone+'-bigwig','X'+str(chrom_set)+'.bw')
        bw = pyBigWig.open(bigwig_file_path)

    retrieve_histones_data(bw, histone_index)

    bw.close()


np.save('/home/vegeta/Downloads/ML4G_Project_1_Data/my_solution/X3.npy', X)





