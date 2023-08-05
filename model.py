import torch
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
import FLAGS
from generate_graphpair import update_anchor_batch
from torch.nn.functional import  normalize, softmax
from torch_geometric.nn.norm.graph_norm import GraphNorm
from torch.nn import Softmax
class TEGConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device=torch.device("cpu")):
        super(TEGConv, self).__init__()
        self.device = device
        self.edge_mlp = torch.nn.Linear(input_dim, output_dim).to(self.device)

    def forward(self, x, edge_index, edge_features):
        source_ids = edge_index[0, :]
        x = x[source_ids]
        x2 = torch.cat([x, edge_features], dim=1)
        x2 = self.edge_mlp(x2)
        x = scatter_mean(x2, edge_index[1, :], dim=0)
        return x


class TMatching(torch.nn.Module):
    def __init__(self, edge_feature_input_dim, edge_feature_embedding_dim, node_embedding_dim, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.edge_feature_mlp = torch.nn.Linear(edge_feature_input_dim, edge_feature_embedding_dim).to(self.device)
        self.tegConvs = torch.nn.ModuleList([TEGConv(FLAGS.MAX_SUM_LEAFS + edge_feature_embedding_dim, node_embedding_dim, self.device)])
        for i in range(FLAGS.N_LAYERS):
            layer = TEGConv(node_embedding_dim + edge_feature_embedding_dim, node_embedding_dim, self.device)
            self.tegConvs.append(layer)
        self.default_act = torch.nn.ReLU()

        self.final_layer1 = torch.nn.Linear(node_embedding_dim, node_embedding_dim).to(self.device)
        self.final_layer2 = torch.nn.Linear(node_embedding_dim, FLAGS.N_TYPES).to(self.device)
        self.sm = Softmax(dim=-1)
        self.graphNorm = GraphNorm(node_embedding_dim).to(self.device)
        print("N_LABELS: ", FLAGS.N_TYPES)


    def forward(self, data: Batch):
        edge_features = data['node', 'to', 'node'].edge_features.to(self.device)
        edge_features = self.edge_feature_mlp(edge_features)
        edge_feature_transformed = self.default_act(edge_features)
        x = data['node'].x.to(self.device)
        edge_index = data['node', 'to', 'node'].edge_index.to(self.device)
        batch_info = data['node'].batch.clone().to(self.device)
        # print(batch_info)
        update_anchor_batch(data['node'].batch, data['node'].anchor)
        batch_info = data['node'].batch.clone().to(self.device)

        for tegConv in self.tegConvs:
            x = tegConv(x, edge_index, edge_feature_transformed)
            x = self.default_act(x)
            # print(x.shape, batch_info.shape)
            x = self.graphNorm(x,batch_info)
            # x = normalize(x)
        # update_anchor_batch(data['node'].batch, data['node'].anchor)
        xbatch = data['node'].batch.to(self.device)
        xout = scatter_mean(x, xbatch, dim=0, dim_size=data.num_graphs * 2)
        xout = xout.reshape(-1).reshape((-1, 2 * FLAGS.NODE_FEATURE_EMBEDDING_DIM))
        xout1 = xout[:, :FLAGS.NODE_FEATURE_EMBEDDING_DIM]
        xout2 = xout[:, FLAGS.NODE_FEATURE_EMBEDDING_DIM:]
        xsub = xout1 - xout2
        scores = self.final_layer2(self.default_act(self.final_layer1(xsub)))
        scores = self.sm(scores)
        return scores.reshape(-1), x, xout1, xout2
