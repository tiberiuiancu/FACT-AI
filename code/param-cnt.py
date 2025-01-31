from utils import load_data
from model import VictimModel
import numpy as np


def count_parameters(model):
    s = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            s += np.prod(parameter.size())
    return s

for dataset in 'pokec_z', 'pokec_n', 'dblp':
    print(dataset)
    g, index_split = load_data(dataset)
    in_dim = g.ndata['feature'].shape[1]
    hid_dim = 128
    out_dim = max(g.ndata['label']).item() + 1
    label = g.ndata['label']
    for m in ['GCN', 'GraphSAGE', 'GAT', 'APPNP', 'SGC']:
        print(m, count_parameters(VictimModel(in_dim, hid_dim, out_dim, 'cpu', name=m).model))
    print('========================')
