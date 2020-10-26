#! /usr/bin/env python
"""
Unsupervised Graph Classification Training via GCN
Author:	Alston
Date: 2020.10.25
"""

import os
import dgl
import dgl.function as fn
import numpy as np
import networkx as nx
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.data import DGLDataset
from torch.utils.data import DataLoader
from dgi import DGI


if __name__ == "__main__":
    glist, label_dict = dgl.load_graphs("dataset/homograph/YouTube/0.bin")
    g = glist[0]

    in_feats = 8
    n_hidden = 20
    n_layers = 1
    dropout = 0
    patience = 20
    n_edges = g.number_of_edges()
    # create DGI model
    dgi = DGI(g,
              in_feats,
              n_hidden,
              n_layers,
              nn.PReLU(n_hidden),
              dropout)
    dgi_optimizer = th.optim.Adam(dgi.parameters(),)

    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []
    for epoch in range(300):
        dgi.train()
        if epoch >= 3:
            t0 = time.time()

        dgi_optimizer.zero_grad()
        loss = dgi(g.ndata['feat'])
        loss.backward()
        dgi_optimizer.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            th.save(dgi.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping!')
            break
        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | ETputs(KTEPS) {:.2f}".format
              (epoch, np.mean(dur), loss.item(), n_edges / np.mean(dur) / 1000))
