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


def data_to_homograph(scenario, graph_id):
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

            # node and edge features
            # src_node_feat = [0]*len(node_types)
            # src_node_feat[node_types.index(src_type)] = 1
            # if node_original_id.index(src_id)+1 > len(node_feats):
            #     node_feats[len(node_feats) : node_original_id.index(src_id)+1] = [[0]*len(node_types)]
            #     node_feats[node_original_id.index(src_id)] = src_node_feat
            # dst_node_feat = [0]*len(node_types)
            # dst_node_feat[node_types.index(dst_type)] = 1
            # if node_original_id.index(dst_id)+1 > len(node_feats):
            #     node_feats[len(node_feats) : node_original_id.index(dst_id)+1] = [[0]*len(node_types)]
            #     node_feats[node_original_id.index(dst_id)] = dst_node_feat
            # edge_feat = [0]*len(edge_types)
            # edge_feat[edge_types.index(edge_type)] = 1
            # edge_feats.append(edge_feat)


    u_ids, v_ids = th.tensor(u), th.tensor(v)
    # node_feats, edge_feats = th.tensor(node_feats), th.tensor(edge_feats)
    g = dgl.graph((u_ids, v_ids), idtype=th.int32)
    bg = dgl.add_reverse_edges(g)

    # g.ndata['type'] = node_feats
    # g.edata['type'] = edge_feats

    """
    print(g)
    # Youtube/0.csv
    [+] Result:
    Graph(num_nodes=8286, num_edges=139951,
    ndata_schemes={'type': Scheme(shape=(8,), dtype=torch.int64)}
    edata_schemes={'type': Scheme(shape=(26,), dtype=torch.int64)})
    """
    return bg


if __name__ == "__main__":
    start_time = time()
    scenario = "YouTube"
    graph_id = 0
    edge_types = ['execve', 'access', 'mmap2', 'open', 'fstat', 'close', 'read', 'stat', 'write',
                  'unlink', 'clone', 'waitpid', 'bind', 'listen', 'chmod', 'connect', 'writev',
                  'recv', 'ftruncate', 'sendmsg', 'send', 'recvmsg', 'accept', 'sendto', 'recvfrom',
                  'truncate']
    node_types = ['process', 'file', 'MAP_ANONYMOUS', 'stdin', 'stdout', 'stderr', 'NA', 'thread']
    g = data_to_homograph(scenario, graph_id)

    # Utilize random walk to generate node features
    print("graph #" + str(graph_id) + " of scenario " + scenario + " has been saved!")
    result = dgl.sampling.random_walk(g, g.nodes(), length=5, restart_prob=0.3)
    node_feats = result[0]
    g.ndata['feat'] = node_feats[:, 1:]
    print(g)
    print(g.ndata['feat'][0:100])
    end_time = time()
    print("Time used: " + str(end_time-start_time))

    # Store homograph locally
    # dgl_graphname = "dataset/dglGraph/" + scenario + "/" + str(graph_id) + ".bin"
    # graph_labels = {"glabel": th.tensor([graph_id])}
    # dgl.save_graphs(dgl_graphname, [g], graph_labels)