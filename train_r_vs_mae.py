from gvp.data import CATHDataset, ProteinGraphDataset, BatchSampler
import torch
import torch.nn as nn
import numpy as np
import torch_geometric as pyg
import torch_cluster
import matplotlib.pyplot as plt
import utils
from pprint import pprint

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_synthetic_cath(dataset):
    """
    takes in a pyg dataset and modifies it
    for graph-level regression as done in
    Di Giovanni-Rusch et al. (2023) for ZINC.
    """
    pass

class SyntheticCATHTransform(pyg.transforms.BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.h = torch.zeros(data.node_s.size(0)).float()
        u, v = utils.get_diameter_nodes(data)
        data.diam_nodes = torch.tensor([[u],[v]])

        hu = torch.rand((1,)).float()
        hv = torch.rand((1,)).float()
        data.h[u] = hu
        data.h[v] = hv

        x_u = data.x[u]
        x_v = data.x[v]
        s = torch.dot(x_u, x_v).view(data.h[u].shape)
        data.y = torch.tanh(hu + hv + s)

        return data

def wire_edge(coords, radius):
    ei = torch_cluster.radius_graph(coords, r=radius)
    return pyg.utils.to_undirected(ei)

transform = SyntheticCATHTransform()
cath = CATHDataset(path="data/cath_data/chain_set.jsonl", splits_path="data/cath_data/chain_set_splits.json")
trainset, valset, testset = map(ProteinGraphDataset, (cath.train, cath.val, cath.test))    
trainset = ProteinGraphDataset(cath.train, transform=transform)
valset = ProteinGraphDataset(cath.val, transform=transform)
testset = ProteinGraphDataset(cath.test, transform=transform)

train_loader = pyg.loader.DataLoader(trainset, batch_size=32, shuffle=False)
val_loader = pyg.loader.DataLoader(valset, batch_size=32, shuffle=False)
test_loader = pyg.loader.DataLoader(testset, batch_size=32, shuffle=False)

for batch in train_loader:
    print (batch)
    break