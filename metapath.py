#! /usr/bin/env python
"""
Get meta-path instances with a given target node and meta-path schema
Author:	Alston
Date:	2020.12.16
"""
import dgl
import torch as th

def get_metapath_instances(g, metapath, target_node):
    import itertools
    # key:etype  value:adj_matrix
    adj_matrix_dic = {}

    # get edge instances regarding each edge type
    for etype in metapath:
        if etype in adj_matrix_dic.keys():
            continue
        else:
            adj = g.adj(etype=etype, scipy_fmt='csr', transpose=True)
            adj_matrix = th.tensor(adj.toarray())
            adj_matrix_dic[etype] = adj_matrix

    # get meta-path instances regarding target node v
    metapath_instances = []
    target_nodes = [target_node]
    for etype in metapath:
        adj_matrix = adj_matrix_dic[etype]
        index_matrix = th.nonzero(adj_matrix, as_tuple=False)
        new_target_nodes = []
        metapath_instances.append([])
        for target_node in target_nodes:
            nonzero_index = []
            for col in index_matrix:
                if col[0] == target_node:
                    nonzero_index.append(col)
            for index in nonzero_index:
                edge_instance_num = adj_matrix[index[0]][index[1]]
                edge_instance = (index[0].item(), index[1].item())
                for i in range(0, edge_instance_num.item()):
                    metapath_instances[-1].append(edge_instance)
                    new_target_nodes.append(index[1].item())
        target_nodes = set(new_target_nodes)

    result = []
    for edge_instances in metapath_instances:
        result.append([])
        for edge_instance in edge_instances:
            result[-1].append(edge_instance[0])

    # save both 2 nodes for the last edge type
    result.append([])
    for edge_instance in metapath_instances[-1]:
        result[-1].append(edge_instance[-1])

    return list(itertools.product(*result))



if __name__ == "__main__":
    # g = dgl.heterograph({
    #     ('A', 'AB', 'B'): ([0, 0, 1, 2], [1, 1, 2, 3]),
    #     ('B', 'BA', 'A'): ([1, 2, 3], [0, 1, 2])
    #     })
    # print(g)
    # metapath = ['AB', 'BA', 'AB']
    # target_node = 1
    # target_type = 'A'

    from data_to_heterograph import data_to_heterograph
    g = data_to_heterograph("YouTube", 1)
    print("[+]The example heterogeneous graph:")
    print(g)
    print('--------------------------------------------------------------------')
    metapath = [('process', 'clone', 'process'), ('process', 'execve', 'file')]
    target_node = 0
    target_type = 'process'

    metapath_instances = get_metapath_instances(g, metapath, target_node)
    print('[+]The meta-path instances  with the meta-path schema "(process, clone, process), (process, execve, file)":')
    print(metapath_instances)
    print('--------------------------------------------------------------------')
