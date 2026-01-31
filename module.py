from itertools import permutations, product
import math
import torch
import copy
import torch.nn as nn
from torch.nn import Parameter

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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