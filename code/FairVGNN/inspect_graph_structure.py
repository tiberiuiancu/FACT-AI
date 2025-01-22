import dgl
import torch

# Load the graph
path = 'dataset/reproducability'
dataset = 'dblp'
glist, _ = dgl.load_graphs(f'{path}/{dataset}.bin')
g = glist[0]

# Get the indices for training, validation, and test sets
idx_train = torch.where(g.ndata['train_index'])[0]
idx_val = torch.where(g.ndata['val_index'])[0]
idx_test = torch.where(g.ndata['test_index'])[0]

# Get the features, labels, and sensitive attributes
features = g.ndata['feature']
labels = g.ndata['label']
sens = g.ndata['sensitive']

# Select a node index to inspect, for example, the first node
node_index = 0

# Print the details of the selected node
print(f"Node Index: {node_index}")
print(f"Features: {features[node_index]}")
print(f"Label: {labels[node_index]}")
print(f"Sensitive Attribute: {sens[node_index]}")
