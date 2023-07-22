import joblib
import torch
from SuchTree import SuchTree
import numpy as np
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

tp_dict = {'frugivory': 0, 'pollination': 1, 'neg': 2}
tp_dict_id = {v : k for k,v in tp_dict.items()}
def dumpEmbedding():
    tMatchingModel = TMatching(FLAGS.EDGE_FEATURE_DIM, FLAGS.EDGE_FEATURE_EMBEDDING_DIM,
                               FLAGS.NODE_FEATURE_EMBEDDING_DIM)
    tMatchingModel.load_state_dict(torch.load("best_model.pkl"))
    # test_dataset = BTDataset(pos_path="positive_pairs_test.json",neg_path="negative_pairs_test.json")
    # test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    train_dataset = BTDataset(pos_path="positive_pairs_train.json", neg_path="negative_pairs_train.json", tp_dict=tp_dict)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    for _, data in enumerate(train_loader):
        scores, x, xout1, xout2 = tMatchingModel(data)
        print(x.shape)
        anchors = get_acc_anchor(data['node'].anchor)
        xps = []
        print(len(anchors), data.num_graphs)
        print(data.name)
        x = x.detach().cpu().numpy()
        lbs = data['label'].v.reshape((-1,3))
        scores = scores.detach().cpu().numpy()
        for i in range(len(anchors) // 2):
            xps.append([x[anchors[2 * i]:anchors[2 * i + 1], :], x[anchors[2 * i + 1]:anchors[2 * i + 2], :],
                        data['leaf'].anchor[i],lbs[i], scores[i],  data.name[i]])

        joblib.dump(xps, "xfeatures.pkl")


def loadx():
    xps = joblib.load("xfeatures.pkl")
    ic = 0
    ti = 2
    for xp in xps:
        f1, f2, anchor_leafs, lb, sc, name = xp
        # print(lb)
        if lb[ti] == 1:

            print(name, anchor_leafs, f1.shape, f2.shape)
            pca = TSNE(n_components=2)
            x = np.vstack((f1,f2))
            xs = pca.fit_transform(x)

            fig = plt.figure()
            x1 = xs[:f1.shape[0],:]
            x2 = xs[f1.shape[0]:, :]
            print(x1.shape, x2.shape)
            c1 = ['red' for _ in range(anchor_leafs[0])] + ['mistyrose' for _ in range(anchor_leafs[0], x1.shape[0])]
            c2 = ['blue' for _ in range(anchor_leafs[1])] + ['lightsteelblue' for _ in range(anchor_leafs[1], x2.shape[0])]
            plt.scatter(x1[:, 0], x1[:, 1], c=c1)
            plt.scatter(x2[:, 0], x2[:, 1], c=c2)
            plt.title(name + "_" + "%s" % tp_dict_id[ti] + "_" + "%s" % sc)
            plt.show()
            # plt.savefig
            # break
            ic += 1
            if ic == 10:
                break




if __name__ == "__main__":
    # dumpEmbedding()
    loadx()
