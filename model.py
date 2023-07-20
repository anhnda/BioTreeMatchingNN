import torch
from torch_scatter import scatter_mean

import FLAGS


class TEGConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TEGConv, self).__init__()
        self.edge_mlp = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_index, edge_features):
        print("Input x: ", x.shape)
        source_ids = edge_index[0, :]
        x = x[source_ids]
        x2 = torch.cat([x, edge_features], dim=1)
        x2 = self.edge_mlp(x2)
        print("X2 shape: ", x2.shape)
        print("Edge_index shape: ", edge_index.shape)
        x = scatter_mean(x2, edge_index[1, :], dim=0)
        return x


class TMatching(torch.nn.Module):
    def __init__(self, edge_feature_input_dim, edge_feature_embedding_dim, node_embedding_dim):
        super().__init__()
        self.edge_feature_mlp = torch.nn.Linear(edge_feature_input_dim, edge_feature_embedding_dim)
        self.tegConvs = torch.nn.ModuleList([
            TEGConv(FLAGS.MAX_SUM_LEAFS + edge_feature_embedding_dim, node_embedding_dim),
            TEGConv(node_embedding_dim + edge_feature_embedding_dim, node_embedding_dim),
            TEGConv(node_embedding_dim + edge_feature_embedding_dim, node_embedding_dim)])
        self.default_act = torch.nn.ReLU()

    def forward(self, x, edge_index, origin_edge_features):
        edge_feature_transformed = self.edge_feature_mlp(origin_edge_features)
        for tegConv in self.tegConvs:
            x = self.default_act(tegConv(x, edge_index, edge_feature_transformed))
        return x
