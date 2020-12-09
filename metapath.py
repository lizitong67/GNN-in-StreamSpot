import dgl
import torch as th
import dgl.function as fn
import numpy as np
import torch.nn as nn
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GATConv
from data_to_homograph import same_direction
from torch.utils.data import *
from torch.utils.data import DataLoader
from dgi import DGI
from anomaly_detection import Autoencoder


def test_graph():
    graph_path =  'dataset/homograph/train/YouTube/3.bin'
    g_list, label_dict = dgl.load_graphs(graph_path)
    graph = g_list[0]
    graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
    return graph


import dgl.convert
g = dgl.heterograph({
    ('A', 'AB', 'B'): ([0, 0, 1, 2], [1, 1, 2, 3]),
    ('B', 'BA', 'A'): ([1, 2, 3], [0, 1, 2])
    })
metapath = ['AB', 'BA', 'AB']
metapath_instances = []
for etype in metapath:
    adj = g.adj(etype=etype, scipy_fmt='csr', transpose=True)
    adj_matrix = th.tensor(adj.toarray())
    edges = th.nonzero(adj_matrix, as_tuple=False)
    for indices in edges:
        src_index, dst_index = indices[0].item(), indices[1].item()
        edge_num = adj_matrix[src_index][dst_index].item()
        metapath_instances.append({etype:(src_index,dst_index)})
        # else:
        #     for instance in metapath_instances:
        #         if instance[-1] == src_index:
        #             instance.append(dst_index)
        #

print(metapath_instances)



