#! /usr/bin/env python
"""
Unsupervised Graph Classification Training via DGI
Author:	Alston
Date: 2020.10.25
"""

import os
import dgl
import numpy as np
import matplotlib.pyplot as plt
import time
import torch as th
import torch.nn as nn
from dgl.data import DGLDataset
from torch.utils.data import *
from torch.utils.data import DataLoader
from dgi import DGI
from anomaly_detection import Autoencoder



# Customized Dataset
class Normal_Dataset(DGLDataset):
    def __init__(self):
        super(Normal_Dataset, self).__init__(name='Normal_Dataset', verbose=True)  # 调用父类构造方法


    def has_cache(self):
        return True

    def load(self):
        self.graph_list = []
        self.label_list = []
        normal_scenario = "dataset/homograph/normal"
        scenarios = os.listdir(normal_scenario)
        for scenario in scenarios:
            file_path = "dataset/homograph/normal/" + scenario
            graphs = os.listdir(file_path)
            for graph in graphs:
                g_list, label_dict = dgl.load_graphs(file_path + '/' + graph)
                self.graph_list.append(g_list[0])
                self.label_list.append(1)

    def __getitem__(self, idx):
        """
         Get graph and label by index
        """
        return self.graph_list[idx], th.tensor(self.label_list[idx])

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graph_list)

class Attack_Dataset(DGLDataset):
    def __init__(self):
        super(Attack_Dataset, self).__init__(name="Attack_Dataset", verbose=True)  # 调用父类构造方法

    def has_cache(self):
        return True

    def load(self):
        self.graph_list = []
        self.label_list = []
        attack_scenario = "dataset/homograph/attack"
        scenarios = os.listdir(attack_scenario)
        for scenario in scenarios:
            file_path = "dataset/homograph/attack/" + scenario
            graphs = os.listdir(file_path)
            for graph in graphs:
                g_list, label_dict = dgl.load_graphs(file_path + '/' + graph)
                self.graph_list.append(g_list[0])
                self.label_list.append(0)

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
    # Training the graph embedding model
    # create DGI model
    dgi = DGI(in_feats,
              n_hidden,
              dropout)
    dgi_optimizer = th.optim.Adam(dgi.parameters())

    for epoch in range(100):
        dgi.train()
        t0 = time.time()
        loss_list = []
        for batched_graph, labels in train_dataloader:
            features = batched_graph.ndata['feat'].float()
            loss = dgi(batched_graph, features)
            dgi_optimizer.zero_grad()
            loss.backward()
            dgi_optimizer.step()
            loss_list.append(loss.item())
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f}".format(epoch, time.time()-t0, np.mean(loss_list)))
    th.save(dgi.state_dict(), 'best_dgi.pkl')
    print("[+] The best graph embedding model has been saved.")

def train_autoencoder():
    dgi = DGI(in_feats,
              n_hidden,
              dropout)
    dgi.load_state_dict(th.load('best_dgi.pkl'))

    autoencoder = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = th.optim.Adam(autoencoder.parameters())

    for epoch in range(30):
        autoencoder.train()
        t0 = time.time()
        loss_list = []
        for batched_graph, labels in train_dataloader:
            features = batched_graph.ndata['feat'].float()
            # get graph embedding
            g_embedding = dgi.encoder(batched_graph, features)
            outputs = autoencoder(g_embedding)
            loss = criterion(outputs, g_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f}".format(epoch, time.time()-t0, np.mean(loss_list)))
    th.save(autoencoder.state_dict(), 'best_autoencoder.pkl')
    print("[+] The best autoencoder classifier model has been saved.")

def classification():
    dgi = DGI(in_feats, n_hidden, dropout)
    classifier = Autoencoder()
    dgi.load_state_dict(th.load('best_dgi.pkl'))
    classifier.load_state_dict(th.load('best_autoencoder.pkl'))
    criterion = nn.MSELoss()
    loss_label = []
    for graphs, labels in test_dataloader:
        features = graphs.ndata['feat'].float()
        g_embedding = dgi.encoder(graphs, features)
        outputs = classifier(g_embedding)
        loss = criterion(outputs, g_embedding)
        # print(loss)
        # tensor(0.0957, grad_fn=<MeanBackward0>)
        loss_label.append([loss.item(), labels.item()])
    print(loss_label)

if __name__ == "__main__":
    # initial parameters
    in_feats = 8
    n_hidden = 20
    dropout = 0
    patience = 20
    batch_size = 16

    # load dataset
    train_split = 0.8
    normal_dataset = Normal_Dataset()
    attack_dataset = Attack_Dataset()
    train_size = int(train_split * len(normal_dataset))
    test_size_normal = len(normal_dataset) - train_size
    train_dataset, test_dataset_normal = random_split(normal_dataset, [train_size, test_size_normal])
    test_dataset = ConcatDataset([attack_dataset, test_dataset_normal])

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
    # train_autoencoder()
    classification()


