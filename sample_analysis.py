from gvp.data import CATHDataset, ProteinGraphDataset, BatchSampler
import torch
import torch.nn as nn
import torch_geometric as pyg

device = "cuda" if torch.cuda.is_available() else "cpu"

cath = CATHDataset(path="data/cath_data/chain_set.jsonl", splits_path="data/cath_data/chain_set_splits.json")
trainset, valset, testset = map(ProteinGraphDataset, (cath.train, cath.val, cath.test))

train_loader = pyg.loader.DataLoader(trainset, batch_size=5, shuffle=False)
val_loader = pyg.loader.DataLoader(valset, batch_size=5, shuffle=False)
test_loader = pyg.loader.DataLoader(testset, batch_size=5, shuffle=False)

for batch in dataloader:
    print (batch)
    break