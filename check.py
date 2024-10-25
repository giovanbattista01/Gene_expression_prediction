import numpy as np
import pyBigWig
import pandas as pd
import matplotlib.pyplot as plt
from histone_data import load_gene_info
from scipy.stats import spearmanr
import time


def main():
    gene_names, chrs, tss_centers, strands, gex = load_gene_info('val',2)
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

    x = range(200000)

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

    unactive_count = 0
    active_count = 0

    count = 0

    for i in range(len(active)):
        try:
            u,a = unactive[i], active[i]
            unactive_data = dnase.values(chrs[u],tss_centers[u]-1000,tss_centers[u]+1000)
            active_data = dnase.values(chrs[a],tss_centers[a]-1000,tss_centers[a]+1000)

            if max(unactive_data) > 1:
                unactive_count += 1

            if max(active_data) > 1:
                active_count += 1

            count +=1

        except:
            continue

    print(active_count/count *100, unactive_count/count *100)
                




    


main()