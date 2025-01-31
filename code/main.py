import argparse

parser = argparse.ArgumentParser(description='Fairness Attack Source code')

parser.add_argument('--dataset', default='pokec_z', choices=['pokec_z','pokec_n','dblp'])

parser.add_argument('--hid_dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--T', type=int, default=20, help='sampling times of Bayesian Network')
parser.add_argument('--theta', type=float, default=0.5, help='bernoulli parameter of Bayesian Network')
parser.add_argument('--node', type=int, default=102, help='budget of injected nodes')
parser.add_argument('--edge', type=int, default=50, help='budget of degrees')
parser.add_argument('--alpha', type=float, default=0.01, help='weight of loss_cf')
parser.add_argument('--beta', type=float, default=4, help='weight of loss_fair')
parser.add_argument('--defense', type=float, default=0, help='the ratio of defense')

parser.add_argument('--ratio', type=float, default=0.5, help='node of top ratio uncertainty are attacked')
parser.add_argument('--before', action='store_true')
parser.add_argument('--models', type=str, nargs='*', default=[])
parser.add_argument('--att_heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--loops', type=int, default=50)

parser.add_argument('--mode', type=str, default='uncertainty', choices=['uncertainty','degree'], help='principle for selecting target nodes')

parser.add_argument('--proxy', type=str, default='direct', choices=['direct','k_hops','pca','k_hops+pca'], help='proxy method for simulating black-box attacks')
parser.add_argument('--k_hops', type=int, default=2, help='number of hops to build the proxy subgraph')
parser.add_argument('--root', type=int, help='start k-hops from this node instead of from the one with the highest degree')
parser.add_argument('--components', type=int, default=8, help='number of principal components to keep')

parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train the victim model')
parser.add_argument('--bn_epochs', type=int, default=500, help='number of epochs to train the bayesian network')
parser.add_argument('--surrogate', type=str, default='GCN', choices=['GCN','GAT'], help='surrogate model architecture for feature attack')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--patience', type=int, default=50, help='early stop patience')
parser.add_argument('--n_times', type=int, default=1, help='times to run')

parser.add_argument('--device', type=int, default=0, help='device ID for GPU')
parser.add_argument('--seed', type=int, help='random seed for reproducibility')
parser.add_argument('--output_path', type=str, help='if set, a csv output is produced at the specified path; the destination folder must exist')

args = parser.parse_args()
print(args)

# -----------------------------------main------------------------------------------ 

from copy import deepcopy
import os
import json

import torch
import numpy as np
import dgl
from codecarbon import EmissionsTracker

from model import VictimModel
from attack import Attacker
from utils import load_data, extract_index_split
import proxy as prx

emissions_export_path = os.getenv('EMISSIONS_EXPORT_PATH')
if emissions_export_path:
    tracker = EmissionsTracker()
    tracker.start()

device = torch.device("cuda", args.device) if torch.cuda.is_available() else torch.device("cpu")

if args.seed is not None:
    torch.manual_seed(args.seed)

if args.before:
    B_ACC = {model:[] for model in args.models}
    B_SP = {model:[] for model in args.models}
    B_EO = {model:[] for model in args.models}
A_ACC = {model:[] for model in args.models}
A_SP = {model:[] for model in args.models}
A_EO = {model:[] for model in args.models}

for i in range(args.n_times):
    g, index_split = load_data(args.dataset)
    g = g.to(device)
    in_dim = g.ndata['feature'].shape[1]
    hid_dim = args.hid_dim
    out_dim = max(g.ndata['label']).item() + 1
    label = g.ndata['label']

    if args.before:
        for model in args.models:
            victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model, args=args)
            victim_model.optimize(g, index_split, args.epochs, args.lr, args.patience)
            acc, sp, eo = victim_model.eval(g, index_split)
            B_ACC[model].append(acc)
            B_SP[model].append(sp)
            B_EO[model].append(eo)

    if args.proxy == 'direct':
        proxy = prx.Direct()
    elif args.proxy == 'k_hops':
        proxy = prx.KHops(args.k_hops, args.root)
    elif args.proxy == 'pca':
        proxy = prx.PCA(args.components)
    elif args.proxy == 'k_hops+pca':
        proxy = prx.Sequential(prx.KHops(args.k_hops, args.root),
                               prx.PCA(args.components))

    g_hat = proxy.approximate(g)

    index_split_hat = extract_index_split(g_hat)
    in_dim_hat = g_hat.ndata['feature'].shape[1]
    
    attacker = Attacker(g_hat, in_dim_hat, hid_dim, out_dim, device, args)
    g_hat_attack, uncertainty_hat = attacker.attack(g_hat, index_split_hat)  # uncertainty shape: [n_nodes]

    g_attack, uncertainty = proxy.reconstruct(g_hat_attack, uncertainty_hat)

    dgl.save_graphs(f'./output/{args.dataset}_poisoned.bin', [g_attack])

    for model in args.models:
        victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model, args=args)
        victim_model.re_optimize(g_attack, uncertainty, index_split, args.epochs, args.lr, args.patience, args.defense)
        acc, sp, eo = victim_model.eval(g_attack, index_split)
        A_ACC[model].append(acc)
        A_SP[model].append(sp)
        A_EO[model].append(eo)

if emissions_export_path:
    tracker.stop()
    results = json.dumps(tracker.final_emissions_data.values) + "\n"
    mode = 'a' if os.path.exists(emissions_export_path) else 'w'
    with open(emissions_export_path, mode) as f:
        f.write(results)

print('================Finished================')
args_dict = deepcopy(vars(args))
args_dict.pop('device')
args_dict.pop('output_path')
args_dict.pop('models')

results = []
str_format = lambda x, y: '{:.2f}±{:.2f}'.format(x, y)
show_output = lambda x, y: print()
for model in args.models:
    print('\033[95m{}\033[0m'.format(model))

    curr = args_dict | {
        'model': model,
        'acc_mean': np.mean(A_ACC[model])*100,
        'acc_std': np.std(A_ACC[model])*100,
        'sp_mean': np.mean(A_SP[model])*100,
        'sp_std': np.std(A_SP[model])*100,
        'eo_mean': np.mean(A_EO[model])*100,
        'eo_std': np.std(A_EO[model])*100,
    }
    print('>> acc:{}'.format(str_format(curr['acc_mean'], curr['acc_std'])))
    print('>> sp:{}'.format(str_format(curr['sp_mean'], curr['sp_std'])))
    print('>> eo:{}'.format(str_format(curr['eo_mean'], curr['eo_std'])))

    if args.before:
        curr |= {
            'before_acc_mean': np.mean(B_ACC[model])*100,
            'before_acc_std': np.std(B_ACC[model])*100,
            'before_sp_mean': np.mean(B_SP[model])*100,
            'before_sp_std': np.std(B_SP[model])*100,
            'before_eo_mean': np.mean(B_EO[model])*100,
            'before_eo_std': np.std(B_EO[model])*100,
        }

        print('>> before acc:'.format(str_format(curr['before_acc_mean'], curr['before_acc_std'])))
        print('>> before sp:{}'.format(str_format(curr['before_sp_mean'], curr['before_sp_std'])))
        print('>> before eo:{}'.format(str_format(curr['before_eo_mean'], curr['before_eo_std'])))

    results.append(curr)


if args.output_path:
    import pandas as pd
    pd.DataFrame(results, columns=list(args_dict.keys()) + [
        'model',
        'before_acc_mean',
        'before_acc_std',
        'before_sp_mean',
        'before_sp_std',
        'before_eo_mean',
        'before_eo_std',
        'acc_mean',
        'acc_std',
        'sp_mean',
        'sp_std',
        'eo_mean',
        'eo_std',
    ]).to_csv(args.output_path, index=False)

    print('Output written to', args.output_path)
