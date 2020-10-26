#! /usr/bin/env python
"""
Graph Classification via GCN
Author:	Alston
Date: 2020.9.14
"""

import os
import dgl
import dgl.function as fn
import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.data import DGLDataset
from torch.utils.data import DataLoader



class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 10)
        self.classify = nn.Linear(10, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            x = F.relu(self.linear1(hg))
            x = F.relu(self.linear2(x))
            output = th.sigmoid(self.classify(x))
            return output

# Customized Dataset
class MyDataset(DGLDataset):
    def __init__(self):
        super(MyDataset, self).__init__(name="MyDataset")

    def process(self):
        self.graph_list = []
        self.label_list = []

        homograph = "dataset/homograph"
        scenarios = os.listdir(homograph)
        for scenario in scenarios:
            filepath = "dataset/homograph/" + scenario
            graphs = os.listdir(filepath)
            for graph in graphs:
                glist, label_dict = dgl.load_graphs(filepath + '/' + graph)
                self.graph_list.append(glist[0])
                for key, value in label_dict.items():
                    if key != 'Drive-by-download':
                        self.label_list.append(0)
                    else:
                        self.label_list.append(1)

    def __getitem__(self, idx):
        """
         Get graph and label by index
        """
        return self.graph_list[idx], th.tensor(self.label_list[idx])

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graph_list)

def collate(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)
    return batched_graph, batched_labels

if __name__ == "__main__":
    dataset = MyDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        collate_fn=collate,
        drop_last=False,
        shuffle=True)
    print(dataloader)

    model = Classifier(8, 20, 2)
    opt = th.optim.Adam(model.parameters())
    for epoch in range(100):
        for batched_graph, labels in dataloader:
            feats = batched_graph.ndata['feat'].float()
            logits = model(batched_graph, feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()     # Clears the gradients of all weights
            loss.backward()     # backward propagation
            opt.step()          # update the weights
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))



"""
Epoch 0 | Loss: 0.6993
Epoch 1 | Loss: 0.7039
Epoch 2 | Loss: 0.6674
Epoch 3 | Loss: 0.6835
Epoch 4 | Loss: 0.6827
Epoch 5 | Loss: 0.6789
Epoch 6 | Loss: 0.6753
Epoch 7 | Loss: 0.6539
Epoch 8 | Loss: 0.6573
Epoch 9 | Loss: 0.6500
Epoch 10 | Loss: 0.6246
Epoch 11 | Loss: 0.6567
Epoch 12 | Loss: 0.6270
Epoch 13 | Loss: 0.6040
Epoch 14 | Loss: 0.6375
Epoch 15 | Loss: 0.6208
Epoch 16 | Loss: 0.6930
Epoch 17 | Loss: 0.6281
Epoch 18 | Loss: 0.5654
Epoch 19 | Loss: 0.6036
Epoch 20 | Loss: 0.5478
Epoch 21 | Loss: 0.5834
Epoch 22 | Loss: 0.6159
Epoch 23 | Loss: 0.5316
Epoch 24 | Loss: 0.5525
Epoch 25 | Loss: 0.4411
Epoch 26 | Loss: 0.4759
Epoch 27 | Loss: 0.4574
Epoch 28 | Loss: 0.3825
Epoch 29 | Loss: 0.5281
Epoch 30 | Loss: 0.3677
Epoch 31 | Loss: 0.4427
Epoch 32 | Loss: 0.3487
Epoch 33 | Loss: 0.3621
Epoch 34 | Loss: 0.3184
Epoch 35 | Loss: 0.3133
Epoch 36 | Loss: 0.3803
Epoch 37 | Loss: 0.2646
Epoch 38 | Loss: 0.2648
Epoch 39 | Loss: 0.2382
Epoch 40 | Loss: 0.2440
Epoch 41 | Loss: 0.2007
Epoch 42 | Loss: 0.1962
Epoch 43 | Loss: 0.2106
Epoch 44 | Loss: 0.2070
Epoch 45 | Loss: 0.4496
Epoch 46 | Loss: 0.1555
Epoch 47 | Loss: 0.1578
Epoch 48 | Loss: 0.1500
Epoch 49 | Loss: 0.3055
Epoch 50 | Loss: 0.1369
Epoch 51 | Loss: 0.1380
Epoch 52 | Loss: 0.1008
Epoch 53 | Loss: 0.1185
Epoch 54 | Loss: 0.1280
Epoch 55 | Loss: 0.1091
Epoch 56 | Loss: 0.0954
Epoch 57 | Loss: 0.1003
Epoch 58 | Loss: 0.0959
Epoch 59 | Loss: 0.4562
Epoch 60 | Loss: 0.2788
Epoch 61 | Loss: 0.0891
Epoch 62 | Loss: 0.0763
Epoch 63 | Loss: 0.1028
Epoch 64 | Loss: 0.2882
Epoch 65 | Loss: 0.0797
Epoch 66 | Loss: 0.2800
Epoch 67 | Loss: 0.0789
Epoch 68 | Loss: 0.0787
Epoch 69 | Loss: 0.0543
Epoch 70 | Loss: 0.2681
Epoch 71 | Loss: 0.0973
Epoch 72 | Loss: 0.0719
Epoch 73 | Loss: 0.0666
Epoch 74 | Loss: 0.0648
Epoch 75 | Loss: 0.0619
Epoch 76 | Loss: 0.0707
Epoch 77 | Loss: 0.0367
Epoch 78 | Loss: 0.0562
Epoch 79 | Loss: 0.0614
Epoch 80 | Loss: 0.0453
Epoch 81 | Loss: 0.0586
Epoch 82 | Loss: 0.0455
Epoch 83 | Loss: 0.0476
Epoch 84 | Loss: 0.0528
Epoch 85 | Loss: 0.0419
Epoch 86 | Loss: 0.2351
Epoch 87 | Loss: 0.0435
Epoch 88 | Loss: 0.0597
Epoch 89 | Loss: 0.0410
Epoch 90 | Loss: 0.2581
Epoch 91 | Loss: 0.2576
Epoch 92 | Loss: 0.0585
Epoch 93 | Loss: 0.0549
Epoch 94 | Loss: 0.0569
Epoch 95 | Loss: 0.2630
Epoch 96 | Loss: 0.2523
Epoch 97 | Loss: 0.0300
Epoch 98 | Loss: 0.0297
Epoch 99 | Loss: 0.0286
Epoch 100 | Loss: 0.0342
"""