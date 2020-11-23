#! /usr/bin/env python
"""
GCN module
Author:	Alston
Date: 2020.10.25
"""
import dgl
import torch as th
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 dropout):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, n_hidden)
        self.conv2 = GraphConv(n_hidden, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))

        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return hg
