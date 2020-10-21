
import dgl
import torch as th
import torch.nn.functional as F

# graph_data = {('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
#               ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
#               ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
#              }
# g = dgl.heterograph(graph_data)
# feats = th.tensor([1,2])
# g.edges[('drug', 'interacts', 'drug')].data["feats"] = feats
# print(g.edges[('drug', 'interacts', 'drug')].data["feats"])
# print(g.etypes)

# node_type_list = ['process', 'thread', 'file', 'MAP_ANONYMOUS', 'NA', 'stdin', 'stdout', 'stderr',
#                   'accept', 'access', 'bind', 'chmod', 'clone', 'close', 'connect', 'execve', 'fstat',
#                   'ftruncate', 'listen', 'mmap2', 'open', 'read', 'recv', 'recvfrom', 'recvmsg', 'send',
#                   'sendmsg', 'sendto', 'stat', 'truncate', 'unlink', 'waitpid', 'write', 'writev']


# import csv
#
# node_types = []
# edge_types = []
# data_path = 'dataset/output_ADM.csv'
# with open(data_path, 'r') as file:
#     reader = csv.reader(file)
#     for line in reader:
#     # data_entry: source-id, source-type, destination-id, destination-type, edge-type, timestamp, graph-id
#         src_type = line[1]
#         des_type = line[3]
#         edge_type = line[4]
#         if src_type not in node_types:
#             node_types.append(src_type)
#         if des_type not in node_types:
#             node_types.append(des_type)
#         if edge_type not in edge_types:
#             edge_types.append(edge_type)
#
# with open('node_type.txt', 'a+') as f:
#     f.write(str(node_types))
# with open('edge_type.txt', 'a+') as f:
#     f.write(str(edge_types))


# l = [[1,2,3], [1,2,2], [4,5,6]]
# ts = th.tensor(l)
# print(ts)

# u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 1, 3, 3])
# g = dgl.graph((u,v), idtype=th.int32)
# g = dgl.add_reverse_edges(g)
# print(bg.edges())
# bg.edata['p'] = th.FloatTensor([1, 1, 1, 1])

# glist, label_dict = dgl.load_graphs("0.bin")
# g = dgl.add_reverse_edges(glist[0])
# result = dgl.sampling.random_walk(g, g.nodes(), length=5, restart_prob=0)
# node_feats = result[0]+1
# node_feats = node_feats.type(th.FloatTensor)
# node_feats = F.normalize(node_feats, p=2, dim=1)
# g.ndata['feat'] = node_feats[:, 1:]

# th.set_printoptions(sci_mode=False)
# print(g.ndata['feat'][0:10])

edge_types = ['execve', 'access', 'mmap2', 'open', 'fstat', 'close', 'read', 'stat', 'write',
              'unlink', 'clone', 'waitpid', 'bind', 'listen', 'chmod', 'connect', 'writev',
              'recv', 'ftruncate', 'sendmsg', 'send', 'recvmsg', 'accept', 'sendto', 'recvfrom',
              'truncate']

import dgl.function as fn
import torch.nn as nn


# u, v = th.tensor([0, 1, 2, 3]), th.tensor([1, 2, 3, 4])
# g = dgl.graph((u,v), idtype=th.int32)
# g.ndata['feat'] = th.ones(5,2)
# g.edata['feat'] = th.ones(4,3)
# print(g.ndata)
# print(g.edata)

# print(g.ndata['feat'])
# node_feat_dim = 2
# linear_src = nn.Parameter(th.FloatTensor(size=(1, node_feat_dim)))
# print(linear_src)
# out_src = g.ndata['feat'] * linear_src
# print(out_src)

# linear_src = nn.Parameter(th.FloatTensor(size=(1, node_feat_dim)))
# linear_dst = nn.Parameter(th.FloatTensor(size=(1, node_feat_dim)))
# out_src = g.ndata['feat'] * linear_src
# out_dst = g.ndata['feat'] * linear_dst
# g.srcdata.update({'out_src': out_src})
# g.dstdata.update({'out_dst': out_dst})
# g.apply_edges(fn.u_add_v('out_src', 'out_dst', 'out'))
# print(g.edata['out'])


# glist, label_dict = dgl.load_graphs("dataset/homograph/YouTube/0.bin")
# g = dgl.add_reverse_edges(glist[0])
# print(g.ndata)
# print(g.edate)


# g1 = dgl.graph(([0, 1], [1, 0]))
# g1.ndata['h'] = th.tensor([1., 2.])
# g2 = dgl.graph(([0, 1], [1, 2]))
# g2.ndata['h'] = th.tensor([1., 2., 3.])
#
#
# print (dgl.readout_nodes(g1, 'h'))
# # tensor([3.])  # 1 + 2
#
# bg = dgl.batch([g1, g2])
# print (dgl.readout_nodes(bg, 'h'))
# # tensor([3., 6.])  # [1 + 2, 1 + 2 + 3]

"""
import os

graph_list = []
label_list = []

homograph = "dataset/homograph"
scenarios = os.listdir(homograph)
for scenario in scenarios:
    filepath = "dataset/homograph/"+scenario
    graphs = os.listdir(filepath)
    for graph in graphs:
        glist, label_dict = dgl.load_graphs(filepath+'/'+graph)
        graph_list.append(glist[0])
        for key, value in label_dict.items():
            if key != 'Attack':
                label_list.append(0)
            else:
                label_list.append(1)
print(len(graph_list))
print(label_list)
"""

# graph_list, label_list = dgl.load_graphs("dataset/homograph/YouTube/0.bin")
u, v = th.tensor([0, 1, 2, 3]), th.tensor([1, 2, 3, 4])
g = dgl.graph((u,v), idtype=th.int32)
g.ndata['feat'] = th.ones(5,2)
g.edata['feat'] = th.ones(4,3)
bg = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
print(bg)