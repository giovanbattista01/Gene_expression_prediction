import numpy as np
import pyBigWig
import pandas as pd
import matplotlib.pyplot as plt
from data import load_gene_info
from scipy.stats import spearmanr
import time
import os
from pathlib import Path
import torch
import torch.nn.functional as F



def main():
    gene_names, chrs, tss_centers, gene_coords , strands, gex = load_gene_info('train',1)
    # gene lengths:  mean: 75000 , median:30000,  std: 140000, max 2M, min 75

    dnase_path = '/home/vegeta/Downloads/ML4G_Project_1_Data/DNase-bigwig/X1.bw'

    dnase = pyBigWig.open(dnase_path)

    i = 0
    gene = dnase.values(chrs[i], gene_coords[i][0] - 2000, gene_coords[i][1] + 2000)

    plt.plot(range(len(gene)),gene)
    d = F.interpolate(torch.tensor(gene).unsqueeze(0).unsqueeze(0), size=20000, mode='linear', align_corners=False).squeeze().detach().numpy()
    #plt.plot(range(len(d)),d)
    plt.show()
    plt.plot(range(len(d)),d)
    plt.show()

    exit()


    # max gex gene index -> 6269
    # min gex gene index -> 0

    """data = np.sort(gex)[:-2000]

    plt.hist(data, bins=20, alpha=0.7, color='blue', edgecolor='black')  # Adjust bins as needed
    plt.title('Histogram of 1D NumPy Array')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()"""

    dnase_path = '/home/vegeta/Downloads/ML4G_Project_1_Data/DNase-bigwig/X1.bw'

    dnase = pyBigWig.open(dnase_path)

    data = (gex > 0).astype(int)

    unactive = np.argwhere(data==0)
    unactive = unactive.reshape(unactive.shape[0])

    active = np.argwhere(data==1)
    active = active.reshape(active.shape[0])

    halfspan  = 50

    x = range(2 * halfspan)

    """for i in range(len(active)):
        try:
            u,a = unactive[i], active[i]
            unactive_data = dnase.values(chrs[u],tss_centers[u]-100000,tss_centers[u]+100000)
            active_data = dnase.values(chrs[a],tss_centers[a]-100000,tss_centers[a]+100000)
            plt.plot(x, active_data, color='red', label='Expressed Gene')  # Red for expressed
            plt.plot(x, unactive_data, color='blue', label='Unexpressed Gene')  # Blue for unexpressed
            
            plt.xlabel('X-axis')
            plt.ylabel('DNase Signal')
            plt.title(f'Gene Comparison {i + 1}: Expressed vs. Unexpressed')
            plt.legend()
            
            # Display the plot for 5 seconds
            plt.show()
            
            # Close the plot to proceed to the next one
            plt.close()
        except:
            continue"""

    """unactive_count = 0
    active_count = 0

    count = 0

    for i in range(len(active)):
        try:
            u,a = unactive[i], active[i]
            unactive_data = dnase.values(chrs[u],tss_centers[u]-50,tss_centers[u]+50)
            active_data = dnase.values(chrs[a],tss_centers[a]-50,tss_centers[a]+50)

            if max(unactive_data) > 0.5:
                unactive_count += 1

            if max(active_data) > 0.5:
                active_count += 1

            count +=1

        except:
            continue

    print(active_count/count *100, unactive_count/count *100)


    simple_preds = np.zeros(len(data))
    for i in range(len(data)):
        try:
            vals = dnase.values(chrs[i],tss_centers[i]-50,tss_centers[i]+50)

            if max(vals) > 0.5:
                simple_preds[i] = 1

        except:
            continue

    print(spearmanr(simple_preds, data), spearmanr(simple_preds,gex), spearmanr(data,gex))"""

    histones_used = ['H3K4me3','H3K4me1','H3K36me3','H3K9me3','H3K27me3']
    colors = ['green','orange','purple','black','blue']

    base_path =  '/home/vegeta/Downloads/ML4G_Project_1_Data/'

    histones = []

    chrom_set = 1

    for i in range(len(histones_used)):

        bigwig_file_path = os.path.join(base_path,histones_used[i]+'-bigwig','X'+str(chrom_set)+'.bigwig')

        if Path(bigwig_file_path).exists():

            bw = pyBigWig.open(bigwig_file_path)

        else:

            bigwig_file_path = os.path.join(base_path,histones_used[i]+'-bigwig','X'+str(chrom_set)+'.bw')
            bw = pyBigWig.open(bigwig_file_path)

        histones.append(bw)


    for i in range(len(active)):
        try:
            a = active[i]
            active_dnase = dnase.values(chrs[a],tss_centers[a]-halfspan,tss_centers[a]+halfspan)
            m,v = np.array(active_dnase).mean(), np.array(active_dnase).std()
            plt.plot(x, active_dnase, color='red', label='Expressed DNAse') 

            histone_data = []
            for j in range(len(histones_used)):
                histone_data = histones[j].values(chrs[a],tss_centers[a]-halfspan,tss_centers[a]+halfspan)
                histone_data = np.array(histone_data)
                hm, hv = histone_data.mean(),histone_data.std()

                histone_data = (histone_data - hm) / hv * v + m

                if j > 9:
                    continue
                
                plt.plot(x, histone_data, color=colors[j], label=histones_used[j]) 

            plt.legend()
            
            # Display the plot for 5 seconds
            plt.show()
            
            # Close the plot to proceed to the next one
            plt.close()
        except Exception as e:
            print(f"An error occurred: {e}")

main()