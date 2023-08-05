from SuchTree import SuchTree
import FLAGS
import numpy as np

from utils import func as utils
import torch
from torch_geometric.data import HeteroData


def convert_suchtree(T: SuchTree, root_offset=0, leaf_offset=0):
    print("A new tree ", root_offset, leaf_offset)
    nl = T.n_leafs
    n = T.length

    x = np.zeros((n, FLAGS.MAX_SUM_LEAFS))
    for i in range(nl):
        x[i, i + leaf_offset] = 1
    remap_node_dict = {}
    sorted_origin_leafs = sorted(list(T.leafs.values()))
    for v in sorted_origin_leafs:
        utils.get_insert_dict_index_with_offset_id(remap_node_dict, v, root_offset)
    edges = []
    edge_labels = []
    current_nodes = sorted_origin_leafs
    next_nodes = set()
    root_node = T.root
    cur_node_level = 0
    node_level_map = dict()

    while True:
        if len(current_nodes) == 0:
            break
        print("START LV: ", cur_node_level, root_node, root_offset)
        if cur_node_level == 0:
            print("LEAF origin: ", current_nodes)
        for current_node_origin in current_nodes:
            assert current_node_origin != root_node
            parent_id_origin = T.get_parent(current_node_origin)

            remap_node_id = utils.get_insert_dict_index_with_offset_id(remap_node_dict, current_node_origin,
                                                                       root_offset)
            remap_parent_id = utils.get_insert_dict_index_with_offset_id(remap_node_dict, parent_id_origin, root_offset)

            if parent_id_origin != root_node:
                assert parent_id_origin not in sorted_origin_leafs
                next_nodes.add(parent_id_origin)
            else:
                node_level_map[remap_parent_id] = FLAGS.ROOT_NODE_LEVEL

            old_level = utils.get_dict(node_level_map, remap_node_id, -10)
            # if old_level < cur_node_level and old_level != 0:
            # print("Old level: ", old_level, cur_node_level)
            assert old_level != 0, "Current lv: %s" % cur_node_level

            if current_node_origin == 36:
                print("TX: ", remap_node_id, current_node_origin, old_level, cur_node_level)
                # exit(-1)
            if remap_node_id == 27:
                print("XID 27", cur_node_level)
            node_level_map[remap_node_id] = cur_node_level

            edges.append([remap_node_id, remap_parent_id])
            if FLAGS.SAME_EDGE_PARENT_CHILD:
                edge_labels.append(1)
            else:
                edge_labels.append(0)
            edges.append([remap_parent_id, remap_node_id])
            edge_labels.append(1)

        current_nodes = sorted(list(next_nodes))
        next_nodes = set()
        cur_node_level += 1

    # Add self-loop
    for i in range(n):
        edges.append([i, i])
        edge_labels.append(1)

    assert len(node_level_map) == n
    # Get Node level list:
    node_level_ar = np.zeros(n, dtype=int)
    node_originid_ar = np.zeros(n, dtype=int)
    remap_x = {v: k for k, v in remap_node_dict.items()}
    children_remap = dict()
    print("REmap x", remap_x)
    print("REmap origin 2 id", remap_node_dict)

    for i in range(n):
        node_level_ar[i] = node_level_map[i + root_offset]
        node_originid_ar[i] = remap_x[i + root_offset]
    for i in range(n):
        if node_level_ar[i] != 0:
            children = T.get_children(node_originid_ar[i])
            children_r = []
            # print("DB i, origin, lv", i, node_originid_ar[i], node_level_ar[i], children, root_offset)
            for child in children:

                if child >= 0:
                    children_r.append(remap_node_dict[child] - root_offset)
            children_remap[i] = children_r
            # for vi in children_r:
            # print("\t CH i, origin, lv", vi, node_originid_ar[vi], node_level_ar[vi])
        else:
            print("Leaf: i, origin, lv ", i + root_offset, node_originid_ar[i], node_level_ar[i])
            pass
    for ni in T.leafs.values():
        ix = remap_node_dict[ni]
        i0 = ix - root_offset
        print("TEST LEAF LV: i, origin, lv: ", ix, ni, node_level_ar[i0])

        if node_level_ar[i0] > 0:
            print(" Check children: ", T.get_children(ni))
    # exit(-1)
    return n, nl, remap_node_dict, edges, edge_labels, x, node_level_ar, node_originid_ar, children_remap


def create_graphpair(T1: SuchTree, T2: SuchTree, links, label=0, to_graph=True, nn=""):
    n1, nl1, remap_node_dict1, edges1, edge_labels1, x1, node_level_ar1, node_origin_ar1, children_remap1 = convert_suchtree(
        T1, 0, 0)
    n2, nl2, remap_node_dict2, edges2, edge_labels2, x2, node_level_ar2, node_origin_ar2, children_remap2 = convert_suchtree(
        T2, n1, nl1)

    # create crossed edges:
    link_rows_names = list(links.index)
    link_columns_names = list(links.columns)
    links = links.to_numpy()
    # print(links, np.any(np.isnan(links)), np.any(np.isinf(links)))

    assert not (np.any(np.isnan(links)) or np.any(np.isinf(links)))
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
                print("NLV: ", j, node_level_ar1[row_remap_ids[j]], node_level_ar2[col_remap_ids[j] -n1])
                assert node_level_ar1[row_remap_ids[j]] == 0
                assert node_level_ar2[col_remap_ids[j] - n1] == 0
                link_edges.append([col_remap_ids[j], row_remap_ids[j]])
                link_edge_data.append([values[j]])
                link_edge_labels.append(2)

                link_edges.append([row_remap_ids[j], col_remap_ids[j]])
                link_edge_data.append([values[j]])
                if FLAGS.SAME_EDGE_HOST_GUEST:
                    link_edge_labels.append(2)
                else:
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
    edge_features[eindices] = xx
    x = torch.from_numpy(np.vstack([x1, x2])).float()

    if to_graph:
        graph = HeteroData()
        graph['node'].x = x
        graph['node', 'to', 'node'].edge_index = edges
        graph['node', 'to', 'node'].edge_features = edge_features
        graph['node'].anchor = [n1, n2]
        graph['leaf'].anchor = [nl1, nl2]
        graph['links'].indices = link_edges
        graph['links'].weight = link_edge_data
        graph['host_node_level'].level = node_level_ar1
        graph['guest_node_level'].level = node_level_ar2
        graph['host_children_map'].dat = [children_remap1]
        graph['guest_children_map'].dat = [children_remap2]
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
