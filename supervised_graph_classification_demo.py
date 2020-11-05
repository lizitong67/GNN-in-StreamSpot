#! /usr/bin/env python
"""
Supervised Graph Classification Training via GCN in YouTube and Attack
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
from torch.utils.data import *


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 10)
        self.classify = nn.Linear(10, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            x = F.relu(self.linear(hg))
            output = th.sigmoid(self.classify(x))
            return output

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
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)
    return batched_graph, batched_labels

def main(train_dataloader, test_dataloader):
    model = Classifier(8, 20, 2)
    opt = th.optim.Adam(model.parameters())
    for epoch in range(80):
        for batched_graph, labels in train_dataloader:
            feats = batched_graph.ndata['feat'].float()
            logits = model(batched_graph, feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()  # Clears the gradients of all weights
            loss.backward()  # backward propagation
            opt.step()  # update the weights
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    th.save(model.state_dict(), 'params.pkl')
    print("[+] The best training model has been saved.")

    # test the saved model
    model.load_state_dict(th.load('params.pkl'))
    correct = 0
    total = 0
    for graphs, labels in test_dataloader:
        feats = graphs.ndata['feat'].float()
        output = model(graphs, feats)
        _, predicted = th.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test accuracy of the model on the test data: {} %'.format(100 * correct / total))

def understand_of_testing(test_dataloader):
    model = Classifier(8, 20, 2)
    model.load_state_dict(th.load('params.pkl'))
    correct = 0
    total = 0
    for graphs, labels in test_dataloader:
        feats = graphs.ndata['feat'].float()
        output = model(graphs, feats)

        print("output: ", end='')
        print(output)
        print("output.data: ", end='')
        print(output.data)
        """
        output: tensor([[0.9746, 0.0305]], grad_fn=<SigmoidBackward>) 
        output.data: tensor([[0.9746, 0.0305]])
        """
        _, predicted = th.max(output.data, 1)  # dim=1; take index as the predicted label: 0 or 1

        print("th.max(output.data, 1): ", end='')
        print(th.max(output.data, 1))
        print(_)
        print(predicted)
        """
        th.max(output.data, 1): torch.return_types.max( values=tensor([0.9746]), indices=tensor([0])) 
        tensor([0.9746]) 
        tensor([0])
        """

        total += labels.size(0)
        # for tensor, bool has the sum() methos; and the item() return the value of tensor(only have one value)
        correct += (predicted == labels).sum().item()
    print('Test accuracy of the model on the test data: {} %'.format(100 * correct / total))

if __name__ == "__main__":
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

    main(train_dataloader, test_dataloader)
    # understand_of_testing(test_dataloader)