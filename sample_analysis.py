from gvp.data import CATHDataset, ProteinGraphDataset
import torch
import torch.nn as nn
import torch_geometric as pyg

device = "cuda" if torch.cuda.is_available() else "cpu"

cath = CATHDataset(path="data/cath_data/chain_set.jsonl", splits_path="data/cath_data/chain_set_splits.json")
trainset, valset, testset = map(ProteinGraphDataset, (cath.train, cath.val, cath.test))

print (type(trainset))
print (len(trainset))
print (trainset[0])
print (trainset[1])