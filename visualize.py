import math

import joblib
import torch
from SuchTree import SuchTree
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import FLAGS
from model import TMatching
import pandas as pd
import seaborn
from generate_graphpair import create_graphpair, update_anchor_batch, get_acc_anchor
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from btdataset import BTDataset
from tqdm import tqdm
from torch.nn import MSELoss
from sklearn.metrics import roc_auc_score as auc, average_precision_score as aupr

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from utils.func import get_scale_light_color_list
from sklearn.linear_model import LinearRegression
tp_dict = {'frugivory': 0, 'pollination': 1, 'neg': 2}
tp_dict_id = {v: k for k, v in tp_dict.items()}
if FLAGS.BINARY_LABEL:
    tp_dict_id = {0: "Pos", 1: "Neg"}
N_LB = len(tp_dict_id)

HOST_COLOR_LV = get_scale_light_color_list('mediumblue')
GUEST_COLOR_LV = get_scale_light_color_list('orangered')
from optparse import OptionParser


def dumpEmbedding():
    tMatchingModel = TMatching(FLAGS.EDGE_FEATURE_DIM, FLAGS.EDGE_FEATURE_EMBEDDING_DIM,
                               FLAGS.NODE_FEATURE_EMBEDDING_DIM)
    tMatchingModel.load_state_dict(torch.load("best_model.pkl"))
    # test_dataset = BTDataset(pos_path="positive_pairs_test.json",neg_path="negative_pairs_test.json")
    # test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    train_dataset = BTDataset(pos_path="positive_pairs_train.json", neg_path="negative_pairs_train.json",
                              tp_dict=tp_dict)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    for _, data in enumerate(train_loader):
        scores, x, xout1, xout2 = tMatchingModel(data)
        print(x.shape)
        anchors = get_acc_anchor(data['node'].anchor)
        links_weight = data['links'].weight
        links_indices = data['links'].indices
        host_children_remap = data['host_children_map'].dat
        guest_children_remap = data['guest_children_map'].dat
        # print(links)
        xps = []
        print(len(anchors), data.num_graphs)
        print(data.name)
        x = x.detach().cpu().numpy()
        lbs = data['label'].v.reshape((-1, N_LB))
        # print(data['host_node_level'].level)
        scores = scores.detach().cpu().numpy().reshape((-1, N_LB))
        for i in range(len(anchors) // 2):
            xps.append([x[anchors[2 * i]:anchors[2 * i + 1], :], x[anchors[2 * i + 1]:anchors[2 * i + 2], :],
                        data['leaf'].anchor[i], lbs[i], scores[i], data['host_node_level'].level[i],
                        data['guest_node_level'].level[i], data.name[i]])



        joblib.dump([xps, links_indices, links_weight, host_children_remap, guest_children_remap], "xfeatures.pkl")

def plotx(ti=0):
    from utils.misc import get_online_position
    xps, links_indices, links_weight, host_children_remap, guest_children_remap = joblib.load("xfeatures.pkl")
    ic = 0
    for ii, xp in enumerate(xps):
        f1, f2, anchor_leafs, lb, sc, host_lv, guest_lv, name = xp
        link_indices = np.asarray(links_indices[ii])
        link_weight = links_weight[ii]
        n_node_host = f1.shape[0]
        n_node_guest = f2.shape[0]
        host_children_remapi = host_children_remap[ii][0]
        guest_children_remapi = guest_children_remap[ii][0]
        # print(lb)
        if lb[ti] == 1:

            print(name, lb, sc, anchor_leafs, f1.shape, f2.shape)
            pca = TSNE(n_components=2)
            x = np.vstack((f1, f2))
            xs = pca.fit_transform(x)

            fig = plt.figure()
            x1 = xs[:f1.shape[0], :]
            x2 = xs[f1.shape[0]:, :]

            # Get leaves
            x1s = []
            x2s = []
            print("SP 12", x1.shape, x2.shape)
            c1 = []
            imap_leaf_host = dict()
            imap_leaf_guest = dict()
            for il in range(x1.shape[0]):
                v = host_lv[il]
                if v == 0:
                    c1.append(HOST_COLOR_LV[v])
                    x1s.append(x1[il])
                    imap_leaf_host[il] = len(x1s) - 1
            c2 = []
            for il in range(x2.shape[0]):
                v = guest_lv[il]
                if v == 0:
                    c2.append(GUEST_COLOR_LV[v])
                    x2s.append(x2[il])
                    imap_leaf_guest[il + n_node_host] = len(x2s) - 1
            x1s = np.vstack(x1s)
            x2s = np.vstack(x2s)

            dat = np.vstack([x1s, x2s])
            online_positions = get_online_position(dat)
            pos_ar_host = []
            pos_ar_guest = []
            print("Imap leaf host: ", imap_leaf_host)
            for i in range(dat.shape[0]):
                if i < x1s.shape[0]:
                    dy = 1
                    color = 'blue'
                else:
                    dy = -1
                    color = 'red'
                dx = online_positions[i]
                if i < x1s.shape[0]:
                    pos_ar_host.append([dx,dy])
                else:
                    pos_ar_guest.append([dx, dy])
                plt.scatter(dx,dy, color=color)
            # mx_weight = -100
            for jj,p in enumerate(link_indices):
                i1, i2 = p
                if i1 > i2:
                    continue
                print(p, x1.shape[0], x2.shape[0])

                i1 = imap_leaf_host[i1]
                i2 = imap_leaf_guest[i2]
                weight = link_weight[jj][0]
                # mx_weight = max(mx_weight, weight)
                # print(weight, mx_weight)
                linewidth = min(10, int(math.ceil(10 * weight)))
                plt.plot([pos_ar_host[i1][0], pos_ar_guest[i2][0]], [pos_ar_host[i1][1], pos_ar_guest[i2][1]],
                         color='green', linewidth=linewidth,linestyle='dashed')

            for parent_id, children in host_children_remapi.items():
                if host_lv[parent_id] == 1:
                    pos = 0
                    nc = 0
                    # print("DB", parent_id, children)
                    for child in children:
                        if child not in imap_leaf_host:
                            continue
                        ichild = imap_leaf_host[child]
                        pos += np.asarray(pos_ar_host[ichild])
                        nc += 1
                    pos /= nc
                    pos[1] += 1
                    plt.scatter(pos[0], pos[1], c='navy')
                    for child in children:
                        if child not in imap_leaf_host:
                            continue
                        ichild = imap_leaf_host[child]
                        child_pos = np.asarray(pos_ar_host[ichild])
                        plt.plot([pos[0], child_pos[0]], [pos[1], child_pos[1]], color='yellow')

            for parent_id, children in guest_children_remapi.items():
                if guest_lv[parent_id] == 1:
                    pos = 0
                    nc = 0
                    # print("DB", parent_id, children)
                    for child in children:
                        if child not in imap_leaf_guest:
                            continue
                        ichild = imap_leaf_guest[child]
                        pos += np.asarray(pos_ar_guest[ichild])
                        nc += 1
                    if nc == 0:
                        continue
                    pos /= nc
                    pos[1] -= 1
                    plt.scatter(pos[0], pos[1], c='brown')
                    for child in children:
                        if child not in imap_leaf_guest:
                            continue
                        ichild = imap_leaf_guest[child]
                        child_pos = np.asarray(pos_ar_guest[ichild])
                        plt.plot([pos[0], child_pos[0]], [pos[1], child_pos[1]], color='yellow')


            plt.title(name + "_" + "%s" % tp_dict_id[ti] + "_" + "%s" % sc[ti])
            plt.show()
            # plt.savefig
            # break
            ic += 1
            if ic == 10:
                break


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-l", "--label", dest="label", type='int', default=0, help="{'frugivory': 0, 'pollination': 1, 'neg': 2}")


    (options, args) = parser.parse_args()
    dumpEmbedding()
    plotx(ti=options.label)
