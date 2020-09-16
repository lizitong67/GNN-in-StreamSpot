#! /usr/bin/env python
"""
GNN demo
Author:	Alston Date:
2020.9.14
"""

import dgl
import dgl.function as fn
import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import HeteroGraphConv
import csv

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_dim, hidden_dim)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hidden_dim, hidden_dim)
            for rel in rel_names}, aggregate='sum')
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feat']
        # Apply graph convolution and activation.
        h = self.conv1(g, h)
        for k, v in h.items():
            h[k] = F.relu(v)
        h = self.conv2(g, h)
        for k, v in h.items():
            h[k] = F.relu(v)

        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            print(hg)
            return self.classify(hg)


glist, label_dict = dgl.load_graphs("dataset/dglGraph/YouTube/0.bin")
g = glist[0]


# embed = nn.Embedding(7065, 5)
# g.nodes['file'].data['feat'] = embed.weight
for node_type in g.ntypes:
    num_nodes = g.num_nodes(node_type)
    embed = nn.Embedding(num_nodes, 5)
    g.nodes[node_type].data['feat'] = embed.weight

rel_names = g.etypes
graph = g
label = th.tensor([1])
model = HeteroClassifier(5, 5, 2, rel_names)
opt = th.optim.Adam(model.parameters())
for epoch in range(20):
    logits = model(graph)
    loss = F.cross_entropy(logits, label)
    opt.zero_grad()
    loss.backward()
    opt.step()

