"""
This script computes the commute time as a function of the radial cutoff used
to create the sparse adjacency for geometric graphs.
"""

from pprint import pprint
import torch
import torch_geometric as tg
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def effective_resistance(edge_index, n_nodes):
    adj = tg.utils.to_dense_adj(edge_index).squeeze(0).view(n_nodes, n_nodes).numpy()
    degree = tg.utils.degree(edge_index[0])
    degree = np.diag(degree.numpy())
    L = degree - adj
    n_ones = np.ones(shape=(n_nodes, n_nodes))
    gamma = np.linalg.pinv(L + ((1/n_nodes) * n_ones))  # moore-penrose inverse
    effres = np.zeros(shape=(n_nodes, n_nodes))

    # print ("Computing eff res ...")
    
    for u in range(n_nodes):
        for v in range(n_nodes):
            res = gamma[u][u] + gamma[v][v] - 2*gamma[u][v]
            effres[u][v] = res

    return effres, gamma # effective resistance matrix R(u,v) for all nodes u, v

def get_commute_time(effres, n_edges):
    return 2 * n_edges * effres # returns a matrix of all the commute times tau(u,v) for all nodes u, v

def radial_cutoff_graph(data, eucl_cutoff):
    # returns pyg-style edge_index with connections based on the radial cutoff
    edge_index = []
    n_nodes = data.h.size(0)
    seen = set()
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and (i,j) not in seen and (j,i) not in seen:
                
                # get coordinates and euclidean distance
                vi = data.x[i]
                vj = data.x[j]
                dist = torch.norm(vi - vj, p=2)
                
                if dist.item() <= eucl_cutoff:
                    edge_index.append([i,j])
                    edge_index.append([j,i])
                seen.add((i,j))
                seen.add((j,i))

    edge_index = torch.tensor(edge_index).long().T
    edge_index = tg.utils.to_undirected(edge_index)
    assert tg.utils.is_undirected(edge_index), "graph not undirected ... fixing."

    return edge_index

def get_diameter_nodes(data):
    data.edge_index = tg.utils.to_undirected(data.edge_index)
    assert tg.utils.is_undirected(data.edge_index), "Adj is not undirected"
    nxg = tg.utils.to_networkx(data)
    nxg = nxg.to_undirected()

    lengths = dict(nx.all_pairs_shortest_path_length(nxg))
    
    max_dist = []
    for i, nbors in lengths.items():
        nbors_list = [[j, d] for j, d in nbors.items()]
        [j_max, d_ij_max] = max(nbors_list, key=lambda rec : rec[1])
        max_dist.append([i, j_max, d_ij_max])

    diam_rec = max(max_dist, key=lambda rec : rec[2])
    i, j = diam_rec[0], diam_rec[1]

    return i, j    