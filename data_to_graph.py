#! /usr/bin/env python
"""
Store the data in DGL heterograph
Author:	Alston
Date:	2020.9.9
"""

import dgl
import dgl.function as fn
import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import csv

# data: source-id, source-type, destination-id, destination-type, edge-type, timestamp, graph-id

def data_to_graph(scenario, graph_id):
    data_path = 'dataset/data/'+scenario+'/'+str(graph_id)+'.csv'
    graph_data = {}
    # description of entity_dic: {entity_1:[4, 8099], entity_1:[2098, 2794], ...}
    entity_dic = {}
    # description of edge_dic: {(src, edge, des):[timestamp, ...], ...}
    edge_dic = {}
    with open(data_path, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            src_id = int(line[0])
            src_type = line[1]
            des_id = int(line[2])
            des_type = line[3]
            edge_type = line[4]
            timestamp = int(line[5])

            # get src_node_id and store node in the entity_dic{}
            if src_type not in entity_dic.keys():
                entity_dic[src_type] = [src_id]
                src_node_id = 0
            else:
                if src_id not in entity_dic[src_type]:
                    entity_dic[src_type].append(src_id)
                # The node id equal to the index of the entity in list
                src_node_id = entity_dic[src_type].index(src_id)

            # get src_node_id and store node in the entity_dic
            if des_type not in entity_dic.keys():
                entity_dic[des_type] = [des_id]
                des_node_id = 0
            else:
                if des_id not in entity_dic[des_type]:
                    entity_dic[des_type].append(des_id)
                # The node_id equal to the index of the entity in list
                des_node_id = entity_dic[des_type].index(des_id)

            # store the node and edge information in graph_data{}
            key = (src_type, edge_type, des_type)
            if key not in graph_data.keys():
                graph_data[key] = [(src_node_id, des_node_id)]
                # store edge feature
                edge_dic[key] = [timestamp]
            else:
                graph_data[key].append((src_node_id, des_node_id))
                # store edge feature
                edge_dic[key].append(timestamp)

    g = dgl.heterograph(graph_data)

    # assign node feature: origin_id(not node_id)
    for entity in entity_dic.keys():
        node_feature = th.tensor(entity_dic[entity])
        g.nodes[entity].data['id'] = node_feature
    # assign edge feature: time
    for edge in edge_dic.keys():
        edge_feature = th.tensor(edge_dic[edge])
        g.edges[edge].data['timestamp'] = edge_feature

    return g
    # print(g)
    # print(g.number_of_nodes('process'))
    # print(g.nodes['process'].data['id'])
    # print(g.edges[('process', 'open', 'file')].data['timestamp'])


if __name__ == '__main__':

    scenario = 'YouTube'
    for graph_id in range(66,100):
        dgl_graph = data_to_graph(scenario, graph_id)
        dgl_graphname = "dataset/dglGraph/"+scenario+"/"+str(graph_id)+".bin"
        graph_labels = {"glabel": th.tensor([graph_id])}
        dgl.save_graphs(dgl_graphname, [dgl_graph], graph_labels)
        print("graph #"+str(graph_id)+" of scenario " + scenario + " has been saved!")


