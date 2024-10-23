from pyfaidx import Fasta
import time
import numpy as np
import h5py


def one_hot_encode(seq):
    mapping = {'a': [1, 0, 0, 0],
               't': [0, 1, 0, 0],
               'c': [0, 0, 1, 0],
               'g': [0, 0, 0, 1],
               'n': [0, 0, 0, 0]}
    return np.array([mapping[base] for base in seq])

def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = {'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}
    return ''.join(complement[base] for base in reversed(seq))


from Bio import SeqIO

def get_chromosome_lengths(fasta_file):
    chromosome_lengths = {}
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        chromosome_lengths[record.id] = len(record.seq)
    
    return chromosome_lengths

hg38_path = '/home/vegeta/Downloads/hg38.fa'


genome = Fasta(hg38_path)


def load_gene_info(mode,chrom_set):   # mode can be train,val or test, chrom_set can be 1,2,3

    base_path = "/home/vegeta/Downloads/ML4G_Project_1_Data/CAGE-train/CAGE-train/"
    file_string = 'X'+str(chrom_set)+'_'+mode
    
    info_path = os.path.join(base_path, file_string + '_info.tsv' )
    y_path = os.path.join(base_path, file_string + '_y.tsv' )

    info = pd.read_csv(info_path, sep='\t')
    y = pd.read_csv(y_path, sep='\t')

    gene_names = info['gene_name']
    chrs = info['chr']
    tss_centers = [ int(( info['TSS_start'][i]+info['TSS_end'][i] ) // 2) for i in range(len(gene_names)) ]
    strands = info['strand']
    gex = y['gex']

    return gene_names, chrs, tss_centers, strands, gex

def pad(seq, mode, amount):
    if mode == 'left':
        return 'n'*amount + seq
    elif mode == 'right':
        return seq + 'n' * amount
    else print('error in padding')



def create_dna_dataset(dataset):

    chrom_lenghts = get_chromosome_lengths(fasta_file)

    for i in range(tss_centers):

        left = max(0, tss_centers[i] - halfspan)
        right = min(tss_centers[i] + halfspan, chrom_lengths[chrs[i]] )

        seq = genome[chrs[i]][left:right].seq

        if left == 0:
            seq = pad(seq, 'left', halfspan - tss_centers[i] )
        
        if right == chrom_lengths[chrs[i]]:
            seq = pad(seq, 'right', tss_centers[i] + halfspan - chrom_lengths[chrs[i]] )

        if strands[i]=='-':
            seq = reverse_complement(seq)

        ohe_seq = one_hot_encode(seq)

        dataset[i] = ohe_seq


gene_names, chroms, tss_centers, strands, gex = load_gene_info('train',1)


SEQ_LENGTH = 200000

NUM_SAMPLES = len(gene_names)

IN_CHANNELS = 4

with h5py.File('X.h5', 'w') as h5file:
    # Create a dataset for one-hot encoded nucleotides as int8
    dataset = h5file.create_dataset(
        'dna_data', 
        (NUM_SAMPLES, SEQ_LENGTH, IN_CHANNELS), 
        dtype='int8'  # Store as int8 for one-hot encoding
    )




