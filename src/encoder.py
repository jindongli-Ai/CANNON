
import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
# from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import reset
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GCNEncoder, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    # def forward(self, x, edge_index, edge_weight=None):
    #     res = []
    #     z = x
    #     res.append(z)
    #     for i, conv in enumerate(self.layers):
    #         z = conv(z, edge_index, edge_weight)
    #         z = self.activation(z)
    #         res.append(z)
    #     tt = res[0]
    #     for i in range(1, len(res)):
    #         tt += res[i]
    #     res.append(tt)
    #     z = torch.cat(res, dim=1)
    #     return z
    
    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()
        
    def forward(self, x, batch, type='mean_pool'):
        if type == 'mean_pool':
            return global_mean_pool(x, batch)
        elif type == 'max_pool':
            return global_max_pool(x, batch)
        elif type == 'sum_pool':
            return global_add_pool(x, batch)


class SubEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, pool, activation, num_layers, device):
        super(SubEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.pool = pool
        self.device = device
        self.conv = GCNEncoder(input_dim, hidden_dim, activation, num_layers)
        self.reset_parameters()
        
    def reset_parameters(self):
        reset(self.conv)
        reset(self.pool)
        
    def forward(self, x, edge_index, subgraphs=None, edge_weight=None, batch=None):
        hidden = self.conv(x, edge_index)
        # summarys = []
        summarys = torch.tensor([]).to(self.device)
        # for subg in subgraphs:
        #     if len(subg) == 1:
        #         summarys = torch.cat((summarys, hidden[torch.tensor(subg)]))
        #         # summarys.append(hidden[torch.tensor(subg)])
        #     else:
        #         summarys = torch.cat((summarys, self.pool(hidden[torch.tensor(subg)], None)))
        #         # summarys.append(self.pool(hidden[torch.tensor(subg)], None))
        # # summarys = torch.tensor([item.cpu().detach().numpy() for item in summarys]).to(self.device)
        return hidden, summarys
    
    
# class RWEncoder(nn.Module):
#     # max-step: max length of walks
#     # hidden_graphs: number of hidden graphs
#     # size-hidden-graphs: number of nodes of each hidden graph
#     # hidden_dim: size of hidden layer
#     # penultimate_dim: size of penultimate layer
#     # normalize: whether to normalize the kernel values
#     def __init__(self, input_dim, max_step, hidden_graphs, size_hidden_graphs, hidden_dim, penultimate_dim, normalize, output_dim, dropout, device):
#         super(RWEncoder, self).__init__()
#         self.max_step = max_step
#         self.hidden_graphs = hidden_graphs
#         self.size_hidden_graphs = size_hidden_graphs
#         self.normalize = normalize
#         self.device = device
#         self.adj_hidden = Parameter(torch.FloatTensor(hidden_graphs, (size_hidden_graphs*(size_hidden_graphs-1))//2))
#         self.features_hidden = Parameter(torch.FloatTensor(hidden_graphs, size_hidden_graphs, hidden_dim))
#         self.fc = torch.nn.Linear(input_dim, hidden_dim)
#         self.bn = nn.BatchNorm1d(hidden_graphs*max_step)
#         self.fc1 = torch.nn.Linear(hidden_graphs*max_step, penultimate_dim)
#         self.fc2 = torch.nn.Linear(penultimate_dim, output_dim)
#         self.dropout = nn.Dropout(p=dropout)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.init_weights()

#     def init_weights(self):
#         self.adj_hidden.data.uniform_(-1, 1)
#         self.features_hidden.data.uniform_(0, 1)
        
#     def forward(self, adj, features, graph_indicator):
#         unique, counts = torch.unique(graph_indicator, return_counts=True)
#         n_graphs = unique.size(0)
#         n_nodes = features.size(0)

#         if self.normalize:
#             norm = counts.unsqueeze(1).repeat(1, self.hidden_graphs)
        
#         adj_hidden_norm = torch.zeros(self.hidden_graphs, self.size_hidden_graphs, self.size_hidden_graphs).to(self.device)
#         idx = torch.triu_indices(self.size_hidden_graphs, self.size_hidden_graphs, 1)
#         adj_hidden_norm[:,idx[0],idx[1]] = self.relu(self.adj_hidden)
#         adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 1, 2)
#         x = self.sigmoid(self.fc(features))
#         z = self.features_hidden
#         zx = torch.einsum("abc,dc->abd", (z, x))
        
#         out = list()
#         for i in range(self.max_step):
#             if i == 0:
#                 eye = torch.eye(self.size_hidden_graphs, device=self.device)
#                 eye = eye.repeat(self.hidden_graphs, 1, 1)              
#                 o = torch.einsum("abc,acd->abd", (eye, z))
#                 t = torch.einsum("abc,dc->abd", (o, x))
#             else:
#                 x = torch.spmm(adj, x)
#                 z = torch.einsum("abc,acd->abd", (adj_hidden_norm, z))
#                 t = torch.einsum("abc,dc->abd", (z, x))
#             t = self.dropout(t)
#             t = torch.mul(zx, t)
#             t = torch.zeros(t.size(0), t.size(1), n_graphs, device=self.device).index_add_(2, graph_indicator, t)
#             t = torch.sum(t, dim=1)
#             t = torch.transpose(t, 0, 1)
#             if self.normalize:
#                 t /= norm
#             out.append(t)
            
#         out = torch.cat(out, dim=1)
#         out = self.bn(out)
#         out = self.relu(self.fc1(out))
#         out = self.dropout(out)
#         out = self.fc2(out)
#         # return F.log_softmax(out, dim=1)
#         return out