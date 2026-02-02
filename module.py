from itertools import permutations, product
import math
import torch
import copy
import torch.nn as nn
from torch.nn import Parameter

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



    def _edge_index_to_adjacency_matrix(self,
                                        edge_index,
                                        edge_weight=None,
                                        num_nodes=100,
                                        no_cuda=False):

        if edge_weight is not None:
            edge_weight = edge_weight.squeeze()
        else:
            edge_weight = torch.ones(
                edge_index.size(1)).cuda() if not no_cuda else torch.ones(
                    edge_index.size(1))
        adj_sparse = torch.sparse_coo_tensor(edge_index,
                                             edge_weight,
                                             size=(num_nodes, num_nodes))
        adj = adj_sparse.to_dense()
        row_sum = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
        d_inv_sqrt_mat = torch.diag_embed(d_inv_sqrt)
        gcn_fact = torch.matmul(d_inv_sqrt_mat,
                                torch.matmul(adj, d_inv_sqrt_mat))

        if not no_cuda and torch.cuda.is_available():
            gcn_fact = gcn_fact.cuda()

        return gcn_fact


    def _heter_no_weight_edge(self, feature, num_modal, dia_lens, win_p,
                              win_f):
        index_inter = []
        all_dia_len = sum(dia_lens)
        all_nodes = list(range(all_dia_len * num_modal))
        nodes_uni = [None] * num_modal

        for m in range(num_modal):
            nodes_uni[m] = all_nodes[m * all_dia_len:(m + 1) * all_dia_len]

        start = 0
        for dia_len in dia_lens:
            for m, n in permutations(range(num_modal), 2):

                for j, node_m in enumerate(nodes_uni[m][start:start +
                                                        dia_len]):
                    if win_p == -1 and win_f == -1:
                        nodes_n = nodes_uni[n][start:start + dia_len]
                    elif win_p == -1:
                        nodes_n = nodes_uni[n][
                            start:min(start + dia_len, start + j + win_f + 1)]
                    elif win_f == -1:
                        nodes_n = nodes_uni[n][max(start, start + j -
                                                   win_p):start + dia_len]
                    else:
                        nodes_n = nodes_uni[n][
                            max(start, start + j -
                                win_p):min(start + dia_len, start + j + win_f +
                                           1)]
                    index_inter.extend(list(product([node_m], nodes_n)))
            start += dia_len
        edge_index = (torch.tensor(index_inter).permute(1, 0).cuda() if
                      not self.no_cuda else torch.tensor(index_inter).permute(
                          1, 0))

        return edge_index

class HeterGConvLayer(torch.nn.Module):

    def __init__(self, feature_size, dropout=0.3, no_cuda=False):
        super(HeterGConvLayer, self).__init__()
        self.no_cuda = no_cuda
        self.hetergconv = SGConv_Our(feature_size, feature_size)

    def forward(self, feature, num_modal, adj_weight):

        if num_modal > 1:
            feature_heter = self.hetergconv(feature, adj_weight)
        else:
            print("Unable to construct heterogeneous graph!")
            feature_heter = feature

        return feature_heter
    

class SGConv_Our(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(SGConv_Our, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output