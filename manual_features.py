def get_features_names():
    list_gene_features = ['gene_len','strand','closeness_to_other_genes']

    signals_list = ['H3K4me3','H3K4me1','H3K36me3','H3K9me3','H3K27me3','DNase']

    thresholds = [1,5,10]

    spans = [50,200,1000]

    list_tss_features = [] 

                                            ### LEGEND ###

    # tss_Peak_-50:50_1  means this is a binary feature that it's 1 if max(signal) around the tss (-50:50) > 1

    for signal in signals_list:
        for t in thresholds:
            for s in spans:
                list_tss_features.append(signal+'_tss_Peak_-'+str(s)+':'+str(s)+'_'+str(t))


    list_additional_features = []

    for signal in signals_list:
        list_additional_features.append(signal + '_max')
        list_additional_features.append(signal + '_min')
        list_additional_features.append(signal + '_max_loc')
        list_additional_features.append(signal + '_min_loc')


    # additionally, let's keep tracks of peaks in the gene area

    for signal in signals_list:
        for t in thresholds:
            list_gene_features.append(signal+'_gene_Peak_'+str(t))


    list_features = list_gene_features + list_tss_features + list_additional_features 

    return list_features




def main():
    print(get_features_names())



main()