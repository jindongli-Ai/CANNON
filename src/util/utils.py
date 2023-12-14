
import time
import networkx as nx
import torch
import dgl
import numpy as np
import scipy.sparse as sp

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 的执行时间为 {execution_time} 秒")
        return result
    return wrapper


def convert_to_dgl_graph(x, edge_index, device):
    dgl_graph = dgl.DGLGraph().to(device)
    dgl_graph.add_nodes(x.shape[0])
    dgl_graph.add_edges(edge_index[0], edge_index[1])
    if x is not None:
        dgl_graph.ndata['feat'] = x
    # if data.edge_attr is not None:
    #     dgl_graph.edata['feat'] = data.edge_attr
    return dgl_graph


@timer_decorator
def generate_rwr_subgraph(dgl_graph, subgraph_size, device):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces, _ = dgl.sampling.random_walk(dgl_graph, all_idx, length=5, restart_prob=0.8)
    subv = []

    for i,trace in enumerate(traces):
        subv.append(torch.unique(trace,sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.sampling.random_walk(dgl_graph, [i], length=5, restart_prob=0.8)
            subv[i] = torch.unique(cur_trace[0], sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 3) and (retry_time > 2):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)
        res_i = []
        for x in subv[i]:
            if x != -1 and x not in res_i:
                res_i.append(x)
        subv[i] = res_i

    return subv

