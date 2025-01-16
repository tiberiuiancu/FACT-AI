import torch
import dgl
import csv
import tqdm
import sys

def progress_bar(it, desc):
    return tqdm.tqdm(
            it,
            desc=f"{desc:>32}",
            disable=None,
            colour='red',
            bar_format='{l_bar}{bar}| {n_fmt:>4}/{total_fmt:>4} [{elapsed}<{remaining}, ' '{rate_fmt}]',
    )

def load_data(dataset):
    dataset = dataset.lower()
    assert dataset in ['pokec_z','pokec_n', 'dblp']
    
    glist, _ = dgl.load_graphs(f'../data/{dataset}.bin')
    g = glist[0]

    return g, extract_index_split(g)

def extract_index_split(g):
    idx_train = torch.where(g.ndata['train_index'])[0]
    idx_val = torch.where(g.ndata['val_index'])[0]
    idx_test = torch.where(g.ndata['test_index'])[0]
    index_split = {'train_index': idx_train,
                   'val_index': idx_val,
                   'test_index': idx_test}
    return index_split

def fair_matrix(pred, label, sens, index):

    SP = []
    EO = []

    idx_d = torch.where(sens[index]==0)[0]
    idx_a = torch.where(sens[index]==1)[0]
    for i in range(label.max()+1):
        # SP
        p_i0 = torch.where(pred[index][idx_d] == i)[0]
        p_i1 = torch.where(pred[index][idx_a] == i)[0]

        sp = (p_i1.shape[0]/idx_a.shape[0]) - (p_i0.shape[0]/idx_d.shape[0])
        SP.append(sp)
        
        # EO
        p_y0 = torch.where(label[index][idx_d] == i)[0]
        p_y1 = torch.where(label[index][idx_a] == i)[0]

        p_iy0 = torch.where(pred[index][idx_d][p_y0] == i)[0]
        p_iy1 = torch.where(pred[index][idx_a][p_y1] == i)[0]

        if p_y0.shape[0] == 0 or p_y1.shape[0] == 0:
            eo = 0
        else:
            eo = (p_iy1.shape[0]/p_y1.shape[0]) - (p_iy0.shape[0]/p_y0.shape[0])
        EO.append(eo)   
    return SP, EO
