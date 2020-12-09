#! /usr/bin/env python
"""
Store the data in DGL homograph
Author:	Alston
Date:	2020.10.8
"""

import dgl
import dgl.function as fn
import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from time import *
import csv

def visualization(g):
    nx_g = g.to_networkx().to_undirected()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.kamada_kawai_layout(nx_g)
    nx.draw(nx_g, pos, with_labels=False, node_color=[[.7, .7, .7]])

def same_direction(scenario, graph_id):
    # The edges of the process reading and writing files are regarded as the same direction
    edge_types = ['execve', 'access', 'mmap2', 'open', 'fstat', 'close', 'read', 'stat', 'write', 'unlink', 'clone',
                  'waitpid', 'bind', 'listen', 'chmod', 'connect', 'writev', 'recv', 'ftruncate', 'sendmsg', 'send',
                  'recvmsg', 'accept', 'sendto', 'recvfrom', 'truncate']
    node_types = ['process', 'file', 'MAP_ANONYMOUS', 'stdin', 'stdout', 'stderr', 'NA', 'thread']

    data_path = 'dataset/split_data/' + scenario + '/' + str(graph_id) + '.csv'
    # data_entry: source-id, source-type, destination-id, destination-type, edge-type, timestamp, graph-id

    # The indexes in the list are node id in graph, and the values are original id in raw data
    node_original_id = []

    # One-hot encoding for node type and edge type
    node_feats, edge_feats = [], []

    # src and des nodes in homograph
    u, v = [], []

    with open(data_path, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            src_id = int(line[0])
            src_type = line[1]
            dst_id = int(line[2])
            dst_type = line[3]
            edge_type = line[4]
            timestamp = int(line[5])
            if src_id not in node_original_id:
                node_original_id.append(src_id)
            u.append(node_original_id.index(src_id))
            if dst_id not in node_original_id:
                node_original_id.append(dst_id)
            v.append(node_original_id.index(dst_id))

            # one-hot encoding for node and edge features
            src_node_feat = [0]*len(node_types)
            src_node_feat[node_types.index(src_type)] = 1
            if node_original_id.index(src_id)+1 > len(node_feats):
                node_feats[len(node_feats) : node_original_id.index(src_id)+1] = [[0]*len(node_types)]
                node_feats[node_original_id.index(src_id)] = src_node_feat
            dst_node_feat = [0]*len(node_types)
            dst_node_feat[node_types.index(dst_type)] = 1
            if node_original_id.index(dst_id)+1 > len(node_feats):
                node_feats[len(node_feats) : node_original_id.index(dst_id)+1] = [[0]*len(node_types)]
                node_feats[node_original_id.index(dst_id)] = dst_node_feat
            edge_feat = [0]*len(edge_types)
            edge_feat[edge_types.index(edge_type)] = 1
            edge_feats.append(edge_feat)

    u_ids, v_ids = th.tensor(u), th.tensor(v)
    node_feats, edge_feats = th.tensor(node_feats), th.tensor(edge_feats)
    g = dgl.graph((u_ids, v_ids), idtype=th.int32)
    g.ndata['feat'] = node_feats
    g.edata['feat'] = edge_feats

    # To eliminate 0-in-degree nodes
    bg = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
    return bg
    # return g

def different_direction(scenario, graph_id):
    # The edges of the process reading and writing files are regarded as the same direction
    edge_types = ['execve', 'access', 'mmap2', 'open', 'fstat', 'close', 'read', 'stat', 'write', 'unlink', 'clone',
                  'waitpid', 'bind', 'listen', 'chmod', 'connect', 'writev', 'recv', 'ftruncate', 'sendmsg', 'send',
                  'recvmsg', 'accept', 'sendto', 'recvfrom', 'truncate']
    node_types = ['process', 'file', 'MAP_ANONYMOUS', 'stdin', 'stdout', 'stderr', 'NA', 'thread']

    data_path = 'dataset/split_data/' + scenario + '/' + str(graph_id) + '.csv'
    # data_entry: source-id, source-type, destination-id, destination-type, edge-type, timestamp, graph-id

    # if edge_types in ['execve', 'access', '']


if __name__ == "__main__":
    start_time = time()
    scenario = "GMail"
    for graph_id in range(100, 101):
        g = same_direction(scenario, graph_id)
        # visualization(g)
        # break
        # # Utilize random walk to generate node features
        # result = dgl.sampling.random_walk(g, g.nodes(), length=5, restart_prob=0)
        # node_feats = result[0] + 1
        # # normalize
        # node_feats = node_feats.type(th.FloatTensor)
        # node_feats = F.normalize(node_feats, p=2, dim=1)
        # g.ndata['feat'] = node_feats[:, 1:]

        # Store homograph locally
        dgl_graphname = "dataset/homograph/normal/" + scenario + "/" + str(graph_id) + ".bin"
        graph_labels = {scenario: th.tensor([graph_id])}
        dgl.save_graphs(dgl_graphname, [g], graph_labels)
        print("graph #" + str(graph_id) + " of scenario " + scenario + " has been saved!")

    end_time = time()
    print("Time used: " + str(end_time - start_time))