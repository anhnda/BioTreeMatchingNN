import torch
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
import FLAGS
from generate_graphpair import update_anchor_batch
from torch.nn.functional import normalize, softmax
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
    def __init__(self, edge_feature_input_dim, edge_feature_embedding_dim, node_embedding_dim,
                 device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.edge_feature_mlp = torch.nn.Linear(edge_feature_input_dim, edge_feature_embedding_dim).to(self.device)
        self.tegConvs = torch.nn.ModuleList(
            [TEGConv(FLAGS.MAX_SUM_LEAFS + edge_feature_embedding_dim, node_embedding_dim, self.device)])
        for i in range(FLAGS.N_LAYERS):
            layer = TEGConv(node_embedding_dim + edge_feature_embedding_dim, node_embedding_dim, self.device)
            self.tegConvs.append(layer)
        self.default_act = torch.nn.ReLU()

        self.final_layer1 = torch.nn.Linear(node_embedding_dim, node_embedding_dim).to(self.device)
        self.final_layer2 = torch.nn.Linear(node_embedding_dim, FLAGS.N_TYPES).to(self.device)
        self.sm = Softmax(dim=-1)
        self.graphNorm = GraphNorm(node_embedding_dim).to(self.device)

        self.host_K = torch.nn.Sequential(
            torch.nn.Linear(FLAGS.NODE_FEATURE_EMBEDDING_DIM, FLAGS.NODE_FEATURE_EMBEDDING_DIM, bias=False),
            torch.nn.ReLU()).to(self.device)
        self.host_Q = torch.nn.Sequential(
            torch.nn.Linear(FLAGS.NODE_FEATURE_EMBEDDING_DIM, FLAGS.NODE_FEATURE_EMBEDDING_DIM, bias=False),
            torch.nn.ReLU()).to(self.device)
        self.host_V = torch.nn.Sequential(
            torch.nn.Linear(FLAGS.NODE_FEATURE_EMBEDDING_DIM, FLAGS.NODE_FEATURE_EMBEDDING_DIM, bias=False),
            ).to(self.device)

        self.guest_K = torch.nn.Sequential(
            torch.nn.Linear(FLAGS.NODE_FEATURE_EMBEDDING_DIM, FLAGS.NODE_FEATURE_EMBEDDING_DIM, bias=False),
            torch.nn.ReLU()).to(self.device)
        self.guest_Q = torch.nn.Sequential(
            torch.nn.Linear(FLAGS.NODE_FEATURE_EMBEDDING_DIM, FLAGS.NODE_FEATURE_EMBEDDING_DIM, bias=False),
            torch.nn.ReLU()).to(self.device)
        self.guest_V = torch.nn.Sequential(
            torch.nn.Linear(FLAGS.NODE_FEATURE_EMBEDDING_DIM, FLAGS.NODE_FEATURE_EMBEDDING_DIM, bias=False),
        ).to(self.device)

        self.last_node_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * FLAGS.NODE_FEATURE_EMBEDDING_DIM, FLAGS.NODE_FEATURE_EMBEDDING_DIM),
            # torch.nn.Dropout(FLAGS.DROP_OUT),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(FLAGS.NODE_FEATURE_EMBEDDING_DIM),
            # torch.nn.Linear(FLAGS.NODE_FEATURE_EMBEDDING_DIM, FLAGS.NODE_FEATURE_EMBEDDING_DIM),
        ).to(self.device)

    def cross_attention(self, q, k, v):
        sim = torch.mm(q, k.t())
        sim = torch.softmax(sim, dim=1)
        attention_x = torch.mm(sim, v)
        return attention_x

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
        batch_anchor = data['node'].anchor
        for i, tegConv in enumerate(self.tegConvs):
            x0 = x
            x = tegConv(x, edge_index, edge_feature_transformed)
            x = self.default_act(x)
            x = self.graphNorm(x, batch_info)
            # Cross mess for i > 0
            if i > 0:
                s = 0
                xcrosses = []
                for n1, n2 in batch_anchor:
                    x_host = x0[s:s + n1]
                    x_guest = x0[s + n1:s + n1 + n2]
                    s += (n1 + n2)
                    attention_host = self.cross_attention(self.host_Q(x_host), self.guest_K(x_guest),
                                                          self.guest_V(x_guest))
                    attention_guest = self.cross_attention(self.guest_Q(x_guest), self.host_K(x_host),
                                                           self.host_V(x_host))
                    xcrosses.append(attention_host)
                    xcrosses.append(attention_guest)
                xcrosses = torch.vstack(xcrosses)
                xcrosses = self.graphNorm(xcrosses, batch_info)

                x = torch.cat([x, xcrosses], dim=1)
                x = self.last_node_mlp(x)
                if FLAGS.JUMP_WEIGHT > 0:
                    x = x0 * FLAGS.JUMP_WEIGHT + x * (1-FLAGS.JUMP_WEIGHT)

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
