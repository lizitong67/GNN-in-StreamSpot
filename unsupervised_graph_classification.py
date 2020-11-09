#! /usr/bin/env python
"""
Unsupervised Graph Classification Training via DGI
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
from torch.utils.data import *
from torch.utils.data import DataLoader
from dgi import DGI, Classifier
from anomaly_detection import Autoencoder

# Customized Dataset
class MyDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="MyDataset")  # 调用父类构造方法

    def has_cache(self):
        return True

    def load(self):
        self.graph_list = []
        self.label_list = []

        homograph = "dataset/homograph"
        scenarios = os.listdir(homograph)
        for scenario in scenarios:
            file_path = "dataset/homograph/" + scenario
            graphs = os.listdir(file_path)
            for graph in graphs:
                g_list, label_dict = dgl.load_graphs(file_path + '/' + graph)
                self.graph_list.append(g_list[0])
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
    # Merge a batch of data
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)
    return batched_graph, batched_labels

def graph_embedding():
    # create DGI model
    dgi = DGI(in_feats,
              n_hidden,
              dropout)
    dgi_optimizer = th.optim.Adam(dgi.parameters())

    # train DGI model for generate the embedding
    dur = []

    for epoch in range(100):
        dgi.train()
        t0 = time.time()
        loss_list = []
        for batched_graph, labels in train_dataloader:
            feats = batched_graph.ndata['feat'].float()
            loss = dgi(batched_graph, feats)
            dgi_optimizer.zero_grad()
            loss.backward()
            dgi_optimizer.step()
            loss_list.append(loss.item())
        avg_loss = np.mean(loss_list)
        dur.append(time.time() - t0)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f}".format
             (epoch, np.mean(dur), avg_loss))
    th.save(dgi.state_dict(), 'best_dgi.pkl')
    print("[+] The best training model has been saved.")

def classify():
    # dataset
    glist, label_dict = dgl.load_graphs("dataset/homograph/YouTube/0.bin")
    g = glist[0]

    # get graph embedding
    dgi = DGI(in_feats,
              n_hidden,
              dropout)
    dgi.load_state_dict(th.load('best_dgi.pkl'))
    feature = g.ndata['feat'].float()
    g_embedding = dgi.encoder(g, feature)

    # anomaly detection
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = th.optim.Adam(model.parameters())

    # continue in from here

if __name__ == "__main__":
    # initial parameters
    in_feats = 8
    n_hidden = 20
    dropout = 0
    patience = 20

    # split dataset
    dataset = MyDataset()
    batch_size = 16
    train_split = 0.8
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        drop_last=False,
        shuffle=True)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate,
        drop_last=False,
        shuffle=True)
    # graph_embedding()
    # classify()

