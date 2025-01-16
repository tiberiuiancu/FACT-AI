import torch
import dgl
import csv
import scipy.sparse as sp
from tqdm import tqdm

def load_data(dataset):
    dataset = dataset.lower()
    assert dataset in ['pokec_z','pokec_n', 'dblp']
    
    glist, _ = dgl.load_graphs(f'data/{dataset}.bin')
    g = glist[0]

    idx_train = torch.where(g.ndata['train_index'])[0]
    idx_val = torch.where(g.ndata['val_index'])[0]
    idx_test = torch.where(g.ndata['test_index'])[0]
    # g.ndata.pop('train_index')
    # g.ndata.pop('val_index')
    # g.ndata.pop('test_index')
    index_split = {'train_index': idx_train,
                    'val_index': idx_val,
                    'test_index': idx_test}
    return g, index_split


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

def compute_heterogeneous_neighbors(g, device):
    """
    Computes heterogeneous neighbors and corresponding features for a graph.
    
    Args:
        g: DGLGraph with 'feature' and 'sensitive' node data.
        device: Torch device (CPU or GPU).
        
    Returns:
        h_X: Feature matrix of heterogeneous neighbors.
        adj_norm_sp: Normalized adjacency matrix (sparse format).
    """
    # Normalize adjacency matrix (remove self-loops)
    adj = g.adj(scipy_fmt="csr") - sp.eye(g.number_of_nodes())
    new_adj = torch.zeros((adj.shape[0], adj.shape[0])).int()

    # Compute heterogeneous neighbors
    for i in tqdm(range(adj.shape[0]), desc="Computing heterogeneous neighbors"):
        neighbors = torch.tensor(adj[i].nonzero()).to(device)
        mask = (g.ndata['sensitive'][neighbors[1]] != g.ndata['sensitive'][i])
        h_nei_idx = neighbors[1][mask]
        new_adj[i, h_nei_idx] = 1

    # Degree normalization and feature aggregation
    deg = np.sum(new_adj.cpu().numpy(), axis=1)
    deg = torch.from_numpy(deg).to(device)
    indices = torch.nonzero(new_adj)
    values = new_adj[indices[:, 0], indices[:, 1]]
    adj_norm_sp = torch.sparse_coo_tensor(indices.t(), values, new_adj.shape).float().to(device)
    h_X = torch.spmm(adj_norm_sp, g.ndata['feature']) / deg.unsqueeze(-1)

    # Handle NaN values
    mask = torch.any(torch.isnan(h_X), dim=1)
    h_X = h_X[~mask].to(device)

    return h_X, adj_norm_sp