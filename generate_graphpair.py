from SuchTree import SuchTree
import FLAGS
import numpy as np

from utils import func as utils
import torch
from torch_geometric.data import HeteroData


def convert_suchtree(T: SuchTree, root_offset=0, leaf_offset=0):
    nl = T.n_leafs
    n = T.length

    x = np.zeros((n, FLAGS.MAX_SUM_LEAFS))
    for i in range(nl):
        x[i, i + leaf_offset] = 1
    remap_node_dict = {}
    sorted_origin_leafs = sorted(list(T.leafs.values()))
    for v in sorted_origin_leafs:
        utils.get_insert_dict_index(remap_node_dict, v, root_offset)
    edges = []
    edge_labels = []
    current_nodes = sorted_origin_leafs
    next_nodes = set()
    root_node = T.root
    while True:
        if len(current_nodes) == 0:
            break
        for current_node_origin in current_nodes:

            parent_id_origin = T.get_parent(current_node_origin)
            if parent_id_origin != root_node:
                next_nodes.add(parent_id_origin)

            remap_node_id = utils.get_insert_dict_index(remap_node_dict, current_node_origin, root_offset)
            remap_parent_id = utils.get_insert_dict_index(remap_node_dict, parent_id_origin, root_offset)
            edges.append([remap_node_id, remap_parent_id])
            edge_labels.append(0)
            edges.append([remap_parent_id, remap_node_id])
            edge_labels.append(1)

        current_nodes = sorted(list(next_nodes))
        next_nodes = set()

    # Add self-loop
    for i in range(n):
        edges.append([i, i])
        edge_labels.append(1)
    return n, nl, remap_node_dict, edges, edge_labels, x


def create_graphpair(T1: SuchTree, T2: SuchTree, links, label=0, to_graph=True, nn=""):
    n1, nl1, remap_node_dict1, edges1, edge_labels1, x1 = convert_suchtree(T1, 0, 0)
    n2, nl2, remap_node_dict2, edges2, edge_labels2, x2 = convert_suchtree(T2, n1, nl1)

    # create crossed edges:
    link_rows_names = list(links.index)
    link_columns_names = list(links.columns)
    links = links.to_numpy()
    # print(links)
    link_mat = links * 1.0 / FLAGS.CROSS_COUNT_NORMALIZE
    row_refs = T1.leafs.keys()
    col_refs = T2.leafs.keys()

    link_edges = []
    link_edge_data = []
    link_edge_labels = []

    remap_colname_matching = dict()
    col_matching_ids = []
    for i, col_name in enumerate(link_columns_names):
        if col_name in col_refs:
            col_matching_ids.append(i)
            remap_colname_matching[len(remap_colname_matching)] = col_name
        else:
            if FLAGS.VERBOSE:
                print("Skip col: ", col_name)
    link_mat = link_mat[:, col_matching_ids]
    for i, row_name in enumerate(link_rows_names):
        if row_name in row_refs:
            counts = link_mat[i]
            nonzero_ids = np.nonzero(counts)[0]
            values = counts[nonzero_ids]

            col_remap_ids = [remap_node_dict2[T2.leafs[remap_colname_matching[j]]] for j in nonzero_ids]
            row_remap_id = remap_node_dict1[T1.leafs[row_name]]
            row_remap_ids = [row_remap_id for _ in range(len(col_remap_ids))]
            for j in range(len(col_remap_ids)):
                link_edges.append([col_remap_ids[j], row_remap_ids[j]])
                link_edge_data.append([values[j]])
                link_edge_labels.append(2)

                link_edges.append([row_remap_ids[j], col_remap_ids[j]])
                link_edge_data.append([values[j]])
                link_edge_labels.append(3)

    edges = edges1 + edges2 + link_edges
    link_edge_started = len(edges1) + len(edges2)
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

    edge_labels = edge_labels1 + edge_labels2 + link_edge_labels
    edge_features = torch.zeros((len(edge_labels), FLAGS.EDGE_FEATURE_DIM))
    eindices = tuple(np.vstack((np.arange(len(edge_labels)), np.asarray(edge_labels, dtype=int))))
    edge_features[eindices] = 1
    eindices = tuple(np.vstack((np.arange(link_edge_started, len(edge_labels)),
                                np.asarray([4 for _ in range(len(link_edge_data))]))))
    xx = torch.from_numpy(np.asarray([link_edge_data])).float().reshape(-1)
    assert not torch.any(torch.isnan(xx))

    edge_features[eindices] = xx
    assert not torch.any(torch.isnan(edge_features))
    assert not torch.any(torch.isnan(edge_features))
    x = torch.from_numpy(np.vstack([x1, x2])).float()

    if to_graph:
        graph = HeteroData()
        graph['node'].x = x
        graph['node', 'to', 'node'].edge_index = edges
        graph['node', 'to', 'node'].edge_features = edge_features
        graph['node'].anchor = [n1, n2]
        graph['leaf'].anchor = [nl1, nl2]
        lbx = torch.zeros(FLAGS.N_TYPES)
        lbx[label] = 1
        graph['label'].v = lbx

        graph.name = nn
        return graph
    return edges, edge_features, x, n1


def update_anchor_batch(batch_tensor, anchor):
    assert np.sum(anchor) == len(batch_tensor)
    s = 0
    cv = 0
    for n1, n2 in anchor:
        batch_tensor[s:s + n1] = cv
        cv += 1
        s += n1
        batch_tensor[s:s + n2] = cv
        cv += 1
        s += n2

def get_acc_anchor(anchor):
    s = 0
    acc_anchor = []
    acc_anchor.append(s)
    for n1, n2 in anchor:
        s += n1
        acc_anchor.append(s)
        s += n2
        acc_anchor.append(s)
    return acc_anchor