import pandas as pd
import numpy as np
from data import load_gene_info
from histone_data import retrieve_histone_data
import os
from pathlib import Path
import pyBigWig
from tqdm import tqdm
import h5py


def get_features_names(list_gene_features, signals_list, thresholds, spans):

    list_tss_features = [] 

                                            ### LEGEND ###

    # tss_Peak_-50:50_1  means this is a binary feature that it's 1 if max(signal) around the tss (-50:50) > 1

    for signal in signals_list:
        for t in thresholds:
            for s in spans:
                list_tss_features.append(signal+'_tss_Peak_-'+str(s)+':'+str(s)+'_'+str(t))
        


    # additionally, let's keep tracks of peaks in the gene area

    for signal in signals_list:
        for t in thresholds:
            list_gene_features.append(signal+'_gene_Peak_'+str(t))
        list_gene_features.append(signal + '_max')
        list_gene_features.append(signal + '_min')
        list_gene_features.append(signal + '_max_loc')
        list_gene_features.append(signal + '_min_loc')


    list_features = list_gene_features + list_tss_features 

    return list_features


def get_features_from_seq(data,gene_index, files_list, gene_names, chroms, tss_centers, gene_coords , strands, gex,signals_list, thresholds, spans):

    data.loc[gene_index,'gene_name']= gene_names[gene_index]
    c =  gene_coords[gene_index]
    strand = int((strands[gene_index]=='+'))
    data.loc[gene_index,'gene_len'] = c[1] - c[0]
    data.loc[gene_index,'strand'] = strand

    lefts = np.array([coord[0] for coord in gene_coords])
    rights = np.array([coord[1] for coord in gene_coords])

    left_distances = c[0] - rights
    right_distances = lefts - c[1]

    l,r = np.inf, np.inf

    try:
        l = np.min(left_distances[left_distances > 0])
        r = np.min(right_distances[right_distances > 0])
    except:
        pass

    d = min(l,r)

    data.loc[gene_index,'closeness_to_other_genes'] = d



    for i,bw_file in enumerate(files_list):

        signal = signals_list[i]
        for t in thresholds:
            # tss features
            for s in spans:
                seq = np.array(bw_file.values(chroms[gene_index],tss_centers[gene_index]-s,tss_centers[gene_index]+s), dtype=np.float32)
                peak = ( np.max(seq) > t ).astype(int)
                data.loc[gene_index, signal+'_tss_Peak_-'+str(s)+':'+str(s)+'_'+str(t)] = peak

            
            # gene features

            seq = np.array(bw_file.values(chroms[gene_index],c[0],c[1]), dtype=np.float32)
            peak = ( np.max(seq) > t ).astype(int)
            data.loc[gene_index,signal+'_gene_Peak_'+str(t)]= peak
            data.loc[gene_index,signal + '_max']= np.max(seq)
            data.loc[gene_index,signal + '_min']= np.min(seq)
            data.loc[gene_index ,signal + '_max_loc'] = np.argmax(seq)
            data.loc[gene_index ,signal + '_min_loc'] = np.argmin(seq)

def build_dataset(mode,cell_line):

    base_dir = "/home/vegeta/Downloads/ML4G_Project_1_Data/"
    save_dir = "/home/vegeta/Downloads/ML4G_Project_1_Data/manual_feature_data/"

    list_gene_features = ['gene_name', 'gene_len','strand','closeness_to_other_genes']

    signals_list = ['H3K4me3','H3K4me1','H3K36me3','H3K9me3','H3K27me3','DNase']

    thresholds = [1,5,10]

    spans = [50,500,5000]

    gene_names, chroms, tss_centers, gene_coords , strands, gex = load_gene_info(mode,cell_line)

    features_names = get_features_names(list_gene_features, signals_list, thresholds, spans)

    data = pd.DataFrame(index=range(len(gene_names)), columns=features_names)

    bw_files = []
    for signal in signals_list:
        bigwig_file_path = os.path.join(base_dir,signal+'-bigwig','X'+str(cell_line)+'.bigwig')

        if Path(bigwig_file_path).exists():

            bw = pyBigWig.open(bigwig_file_path)

        else:

            bigwig_file_path = os.path.join(base_dir,signal+'-bigwig','X'+str(cell_line)+'.bw')
            bw = pyBigWig.open(bigwig_file_path)
        bw_files.append(bw)

    for gene_index in tqdm(range(len(gene_names))):
        get_features_from_seq(data,gene_index, bw_files, gene_names, chroms, tss_centers, gene_coords , strands, gex, signals_list, thresholds, spans)

    data.to_hdf(save_dir+'X'+str(cell_line)+'_'+mode+'.h5', key='data', mode='w')

def main():
    
    build_dataset('train',1)



main()