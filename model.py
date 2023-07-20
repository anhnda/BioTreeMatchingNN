import torch
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
import FLAGS
from generate_graphpair import update_anchor_batch
from torch.nn.functional import  normalize
class TEGConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TEGConv, self).__init__()
        self.edge_mlp = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x, edge_index, edge_features):
        source_ids = edge_index[0, :]
        x = x[source_ids]
        x2 = torch.cat([x, edge_features], dim=1)
        x2 = self.edge_mlp(x2)
        x = scatter_mean(x2, edge_index[1, :], dim=0)
        return x


class TMatching(torch.nn.Module):
    def __init__(self, edge_feature_input_dim, edge_feature_embedding_dim, node_embedding_dim):
        super().__init__()
        self.edge_feature_mlp = torch.nn.Linear(edge_feature_input_dim, edge_feature_embedding_dim)
        self.tegConvs = torch.nn.ModuleList([TEGConv(FLAGS.MAX_SUM_LEAFS + edge_feature_embedding_dim, node_embedding_dim)])
        for i in range(FLAGS.N_LAYERS):
            layer = TEGConv(node_embedding_dim + edge_feature_embedding_dim, node_embedding_dim)
            self.tegConvs.append(layer)
        self.default_act = torch.nn.ReLU()

        self.final_layer1 = torch.nn.Linear(node_embedding_dim, node_embedding_dim)
        self.final_layer2 = torch.nn.Linear(node_embedding_dim, 1)



    def forward(self, data: Batch):
        edge_features = data['node', 'to', 'node'].edge_features
        edge_features = self.edge_feature_mlp(edge_features)
        edge_feature_transformed = self.default_act(edge_features)
        x = data['node'].x

        for tegConv in self.tegConvs:
            x = tegConv(x, data['node', 'to', 'node'].edge_index, edge_feature_transformed)
            x = self.default_act(x)
        update_anchor_batch(data['node'].batch, data['node'].anchor)
        xout = scatter_mean(x, data['node'].batch, dim=0, dim_size=data.num_graphs * 2)
        xout = xout.reshape(-1).reshape((-1, 2 * FLAGS.NODE_FEATURE_EMBEDDING_DIM))
        xout1 = xout[:, :FLAGS.NODE_FEATURE_EMBEDDING_DIM]
        xout2 = xout[:, FLAGS.NODE_FEATURE_EMBEDDING_DIM:]
        xsub = xout1 - xout2
        scores = self.final_layer2(self.default_act(self.final_layer1(xsub)))
        return scores.reshape(-1), x, xout1, xout2
