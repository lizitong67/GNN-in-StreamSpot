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


# Customized Train Dataset
# Load from disk
class Train_Dataset(DGLDataset):
    def __init__(self):
        self.graphs_path_list = []
        self.label_list = []
        super(Train_Dataset, self).__init__(name='Train_Dataset', verbose=False)  # 调用父类构造方法

    def has_cache(self):
        return True

    def load(self):
        normal_scenario = "dataset/homograph/train"
        scenarios = os.listdir(normal_scenario)
        # scenarios = ['YouTube', 'CNN']
        for scenario in scenarios:
            file_path = normal_scenario + '/' + scenario
            graphs = os.listdir(file_path)
            graphs_path = [file_path + '/' + graph for graph in graphs]
            self.graphs_path_list += graphs_path
        self.it_graphs_path_list = iter(self.graphs_path_list)

    def __getitem__(self, idx):
        """
         Get graph and label
        """
        g_list, label_dict = dgl.load_graphs(next(self.it_graphs_path_list))
        return g_list[0], th.tensor([1])

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs_path_list)

# Customized Test Dataset
# load from memory
class Test_Dataset(DGLDataset):
    def __init__(self):
        super(Test_Dataset, self).__init__(name="Test_Dataset", verbose=False)  # 调用父类构造方法

    def has_cache(self):
        return True

    def load(self):
        self.graph_list = []
        self.label_list = []
        path = "dataset/homograph/test"
        scenarios = os.listdir(path)
        for scenario in scenarios:
            file_path =  path + '/' + scenario
            graphs = os.listdir(file_path)
            for graph in graphs:
                g_list, label_dict = dgl.load_graphs(file_path + '/' + graph)
                self.graph_list.append(g_list[0])
                if scenario == 'normal':
                    self.label_list.append(1)
                else:
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

def load_data(data):
    if data == 'train':
        train_dataset = Train_Dataset()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate,
            drop_last=False,
            shuffle=True)
        return train_dataloader
    if data == 'test':
        test_dataset = Test_Dataset()
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=collate,
            drop_last=False,
            shuffle=True)
        return test_dataloader

def graph_embedding():
    # Training the graph embedding model
    # create DGI model
    dgi = DGI(in_feats,
              n_hidden,
              dropout)
    dgi_optimizer = th.optim.Adam(dgi.parameters())

    for epoch in range(60):
        dgi.train()
        t0 = time.time()
        loss_list = []
        train_dataloader = load_data('train')
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

    for epoch in range(8):
        autoencoder.train()
        t0 = time.time()
        loss_list = []
        train_dataloader = load_data('train')
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
    test_dataloader = load_data('test')
    for graphs, labels in test_dataloader:
        features = graphs.ndata['feat'].float()
        g_embedding = dgi.encoder(graphs, features)
        outputs = classifier(g_embedding)
        loss = criterion(outputs, g_embedding)
        # print(loss)
        # tensor(0.0957, grad_fn=<MeanBackward0>)
        loss_label.append([loss, labels])
    print(loss_label)
    # Acc
    for threshold in th.arange(0.1, 0.3, 0.01):
        predicts = []
        labels = []
        for item in loss_label:
            abnormal_score = item[0]
            label = item[1]
            predict = 0 if abnormal_score > threshold else 1
            labels.append(label)
            predicts.append(predict)
        tensor_predicts = th.tensor(predicts)
        tensor_labels = th.tensor(labels)
        correct_num = th.sum(tensor_predicts == tensor_labels)
        correct = correct_num.item() * 1.0 / len(labels)
        print("Threshold: "+str(round(threshold.item(), 3))+"; Acc: "+str(round(correct, 4)))

if __name__ == "__main__":
    # initial parameters
    in_feats = 8
    n_hidden = 20
    dropout = 0
    patience = 20
    batch_size = 16

    # graph_embedding()
    # train_autoencoder()
    classification()

    # train = Train_Dataset()
    # print(len(train))
    # test = Test_Dataset()
    # print(len(test))