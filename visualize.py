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



        joblib.dump(xps, "xfeatures.pkl")


def loadx(ti=0):
    xps = joblib.load("xfeatures.pkl")
    ic = 0
    for xp in xps:
        f1, f2, anchor_leafs, lb, sc, host_lv, guest_lv, name = xp
        # print(lb)
        if lb[ti] == 1:

            print(name, lb, sc, anchor_leafs, f1.shape, f2.shape)
            pca = TSNE(n_components=2)
            x = np.vstack((f1, f2))
            xs = pca.fit_transform(x)

            fig = plt.figure()
            x1 = xs[:f1.shape[0], :]
            x2 = xs[f1.shape[0]:, :]
            x1s = []
            x2s = []
            print(x1.shape, x2.shape)
            c1 = []
            for il in range(x1.shape[0]):
                v = host_lv[il]
                if v != 0:
                    continue
                if v == FLAGS.ROOT_NODE_LEVEL:
                    c1.append((1, 20.0 / 255, 147.0 / 255))  # PINK
                else:
                    v = min(v, len(HOST_COLOR_LV) - 1)
                    c1.append(HOST_COLOR_LV[v])
                x1s.append(x1[il])
            c2 = []
            for il in range(x2.shape[0]):
                v = guest_lv[il]
                if v != 0:
                    continue
                if v == FLAGS.ROOT_NODE_LEVEL:
                    c2.append((0, 1, 0))  # GREEN
                else:
                    v = min(v, len(GUEST_COLOR_LV) - 1)
                    c2.append(GUEST_COLOR_LV[v])
                x2s.append(x2[il])
            # c1 = ['red' for _ in range(anchor_leafs[0])] + [(0.25, 0.0676470588235294, 0.0) for _ in range(anchor_leafs[0], x1.shape[0])]
            # c2 = ['blue' for _ in range(anchor_leafs[1])] + ['lightsteelblue' for _ in range(anchor_leafs[1], x2.shape[0])]
            x1s = np.vstack(x1s)
            x2s = np.vstack(x2s)
            sc1 = plt.scatter(x1s[:, 0], x1s[:, 1], c=c1)
            sc2 = plt.scatter(x2s[:, 0], x2s[:, 1], c=c2)

            dat = np.vstack([x1s, x2s])
            reg = LinearRegression().fit(dat[:,0].reshape(-1,1), dat[:, 1])
            coef = reg.coef_
            intercept = reg.intercept_
            x = np.linspace(np.min(dat[:,0]), np.max(dat[:,0]), 100)
            y = x * coef + intercept
            plt.plot(x,y)
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
    # dumpEmbedding()
    loadx(ti=options.label)
