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
import csv

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, feat):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, feat))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)

import dgl.data
dataset = dgl.data.GINDataset('MUTAG', False)

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)
    return batched_graph, batched_labels

from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=1024,
    collate_fn=collate,
    drop_last=False,
    shuffle=True)

model = Classifier(10, 20, 5)
opt = th.optim.Adam(model.parameters())
for epoch in range(20):
    for batched_graph, labels in dataloader:
        feats = batched_graph.ndata['feats']
        logits = model(batched_graph, feats)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))