import argparse

DATASETS = ['pokec_z', 'pokec_n', 'dblp']

parser = argparse.ArgumentParser(description='Dataset inspector')

parser.add_argument('--color', default='auto', choices=['never','auto','always'], help="display colored output")
parser.add_argument('--datasets', nargs='+', choices=DATASETS, help="names of datasets to inspect")
parser.add_argument('--files', nargs='+', help="paths of datasets to inspect")

args = parser.parse_args()

# -----------------------------------main------------------------------------------

import sys
import dgl
import torch
from sklearn.decomposition import PCA

if args.color == 'always' or args.color == 'auto' and sys.stdout.isatty():
    def color(c='reset'):
        return {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
        }[c]
else:
    def color(c='reset'):
        return ""

datasets = []

if args.files is not None:
    for file in args.files:
        datasets.append((None, file))
elif args.datasets is None:
    args.datasets = DATASETS

if args.datasets is not None:
    for dataset in args.datasets:
        datasets.append((dataset, f"data/{dataset}.bin"))


head = lambda name, path: print(f"{color('bold')}[{path if name is None else name}]{color()}")
prop = lambda key, val: print(f"  {color('bold')}{key}:{color()} {val}")
load = lambda path: dgl.load_graphs(path)[0][0]

def none_all_some(n, N):
    if n == 0:
        return f"{color('red')}none{color()}"
    elif n == N:
        return f"{color('green')}all{color()}"
    else:
        return f"{color('yellow')}some ({n}/{N}){color()}"

def subprops(xy):
    return "".join([f"\n    {color('bold')}{x}:{color()} {y}" for x, y in xy])

def sensitive_nodes(graph):
    sensitive = graph.ndata['sensitive']
    label = graph.ndata['label']

    s0y0 = torch.logical_and(sensitive == 0, label == 0).sum()
    s0y1 = torch.logical_and(sensitive == 0, label == 1).sum()
    s0y_ = torch.logical_and(sensitive == 0, label  < 0).sum()
    s1y0 = torch.logical_and(sensitive == 1, label == 0).sum()
    s1y1 = torch.logical_and(sensitive == 1, label == 1).sum()
    s1y_ = torch.logical_and(sensitive == 1, label  < 0).sum()
    s_y0 = torch.logical_and(sensitive  < 0, label == 0).sum()
    s_y1 = torch.logical_and(sensitive  < 0, label == 1).sum()
    s_y_ = torch.logical_and(sensitive  < 0, label  < 0).sum()

    xy = [
        ("s = 0", s0y0 + s0y1 + s0y_),
        ("s = 0, y = 0", s0y0),
        ("s = 0, y = 1", s0y1),
        ("s = 0, y = ?", s0y_),
        ("s = 1", s1y0 + s1y1 + s1y_),
        ("s = 1, y = 0", s1y0),
        ("s = 1, y = 1", s1y1),
        ("s = 1, y = ?", s1y_),
        ("s = ?", s_y0 + s_y1 + s_y_),
        ("s = ?, y = 0", s_y0),
        ("s = ?, y = 1", s_y1),
        ("s = ?, y = ?", s_y_),
    ]

    return subprops(xy)

def explained_variance(features):
    pca = PCA().fit(features.cpu())
    ev = pca.explained_variance_
    xy = [
        ("by first principal component", ev[0]),
        ("by first 2 principal components", ev[:2].sum()),
        ("by first 4 principal components", ev[:4].sum()),
        ("by first 8 principal components", ev[:8].sum()),
        ("by first 16 principal components", ev[:16].sum()),
        ("by first 32 principal components", ev[:32].sum()),
        ("by first 64 principal components", ev[:64].sum()),
        ("by first 128 principal components", ev[:128].sum()),
    ]
    total = ev.sum()

    xy = [(x, f"{y} ({y/total*100:.1f}%)") for x, y in xy]
    xy.append(("total", ev.sum()))

    return subprops(xy)

def self_connections(graph):
    nodes = torch.arange(graph.num_nodes(), device=graph.device)
    connected = sum(graph.has_edges_between(nodes, nodes))

    return none_all_some(connected, len(nodes))

def bidirectional(graph):
    nodes = torch.arange(graph.num_nodes(), device=graph.device)
    u, v = graph.edges()
    connected = sum(graph.has_edges_between(v, u))

    return none_all_some(connected, len(u))

def khop(graph):
    degree = graph.in_degrees() + graph.out_degrees()
    root = torch.argmax(degree)
    N = graph.num_nodes()

    xy = []

    for k in range(1, 6):
        subgraph, _ = dgl.khop_in_subgraph(graph, root, k + 1)
        n = subgraph.num_nodes()
        xy.append((f"in {k + 1} hop{['','s'][k>0]}", f"{n} nodes ({n/N*100:.1f}%)"))

    return subprops(xy)

def quantiles(data):
    data = data.float()
    x = torch.tensor([0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1])
    p = [f"{int(p)}%" for p in torch.round(x * 100)]
    y = torch.quantile(data, x)

    return subprops(zip(p, y))

for name, path in datasets:
    graph = load(path)

    head(name, path)
    prop("number of nodes", graph.num_nodes())
    prop("number of training nodes", sum(graph.ndata['train_index'].int()))
    prop("number of validation nodes", sum(graph.ndata['val_index'].int()))
    prop("number of test nodes", sum(graph.ndata['test_index'].int()))
    prop("number of sensitive nodes", sensitive_nodes(graph))

    prop("feature dimensionality", graph.ndata['feature'].shape[1])
    prop("explained variance", explained_variance(graph.ndata['feature']))

    prop("number of edges", graph.num_edges())
    prop("self connections", self_connections(graph))
    prop("bidirectional", bidirectional(graph))
    prop("reachable from highest-degree node", khop(graph))
    prop("in degrees", quantiles(graph.in_degrees()))
    prop("out degrees", quantiles(graph.out_degrees()))
