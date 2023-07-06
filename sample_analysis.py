from gvp.data import CATHDataset, ProteinGraphDataset, BatchSampler
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_cluster
import matplotlib.pyplot as plt
import utils
from pprint import pprint

device = "cuda" if torch.cuda.is_available() else "cpu"

cath = CATHDataset(path="data/cath_data/chain_set.jsonl", splits_path="data/cath_data/chain_set_splits.json")
trainset, valset, testset = map(ProteinGraphDataset, (cath.train, cath.val, cath.test))

def create_synthetic_cath(dataset):
    """
    takes in a pyg dataset and modifies it
    for graph-level regression as done in
    Di Giovanni-Rusch et al. (2023) for ZINC.
    """
    pass

def wire_edge(coords, radius):
    return torch_cluster.radius_graph(coords, r=radius)

radii = [i + 2.5 for i in range(2, 29, 2)]
N_graphs = 6
sample_ids = torch.randint(len(trainset), size=(N_graphs,)).view(-1).numpy().tolist()
print (sample_ids)
all_commute_times = []
n_atoms = []

for sid in sample_ids:
    data = trainset[sid]
    commute_times = []
    n_atoms.append(data.x.size(0))
    for r in radii:
        print (sid, r)
        new_edge_index = wire_edge(data.x, r)
        new_edge_index = pyg.utils.to_undirected(new_edge_index)
        data.edge_index = new_edge_index

        u, v = utils.get_diameter_nodes(data)

        effres = utils.effective_resistance(data.edge_index, data.x.size(0))
        ct_mat = utils.get_commute_time(effres, data.edge_index.size(1) // 2)

        ct = ct_mat[u][v]
        commute_times.append(ct)
    all_commute_times.append(commute_times)

print(np.array(all_commute_times).shape)

for i in range(N_graphs):
    assert len(radii) == len(all_commute_times[i])
    plt.plot(radii, all_commute_times[i], label=f"id: {sample_ids[i]} | N: {n_atoms[i]}")
plt.legend()
plt.grid()
plt.xlabel("radial cutoff $r$")
plt.ylabel("commute time $\\tau(u,v)$")
plt.savefig("r_vs_ct.pdf")

# train_loader = pyg.loader.DataLoader(trainset, batch_size=5, shuffle=False)
# val_loader = pyg.loader.DataLoader(valset, batch_size=5, shuffle=False)
# test_loader = pyg.loader.DataLoader(testset, batch_size=5, shuffle=False)

# for i, batch in enumerate(train_loader):
#     print (batch)
#     break