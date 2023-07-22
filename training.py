import random

import torch
from SuchTree import SuchTree
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

import FLAGS
from model import TMatching
import pandas as pd
import seaborn
from generate_graphpair import create_graphpair, update_anchor_batch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from btdataset import BTDataset
from tqdm import tqdm
from torch.nn import MSELoss
from sklearn.metrics import roc_auc_score as auc, average_precision_score as aupr


def train():
    tp_dict = dict()
    train_dataset = BTDataset(pos_path="positive_pairs_train.json",neg_path="negative_pairs_train.json", tp_dict=tp_dict)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.BATCH_SIZE, shuffle=True)
    test_dataset = BTDataset(pos_path="positive_pairs_test.json",neg_path="negative_pairs_test.json", tp_dict=tp_dict)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.BATCH_SIZE, shuffle=False)
    # if not torch.backends.mps.is_available():
    #     device = torch.device("cpu")
    # else:
    device = torch.device("cpu")
    tMatchingModel = TMatching(FLAGS.EDGE_FEATURE_DIM, FLAGS.EDGE_FEATURE_EMBEDDING_DIM,
                               FLAGS.NODE_FEATURE_EMBEDDING_DIM, device).to(device)
    lossFunc = MSELoss()
    optimizer = torch.optim.Adam(tMatchingModel.parameters(), lr=0.001, weight_decay=0.001) # lr=args.lr, weight_decay=args.w_decay
    tMatchingModel.train()
    best_eval_lost = 1000
    ibest = -1
    eval_label = None
    best_predicted_eval_label = None
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(FLAGS.N_EPOCHES):
        tMatchingModel.train()
        for data in tqdm(train_loader, total=len(train_loader), desc="Epoch: %5s" % epoch):
        # for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            scores, x, xout1, xout2 = tMatchingModel(data)
            # print("Predicted: ", scores)
            labels = data['label'].v.float().reshape(-1).to(device)
            # print("Target", labels)
            loss = lossFunc(scores, labels)
            # print(epoch, "Loss: ", loss.data, "Predicted: ", scores, "Target", labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(tMatchingModel.parameters(), max_norm=1, norm_type=2,
            #                                error_if_nonfinite=True)
            optimizer.step()

        tMatchingModel.eval()
        all_predicted = []
        all_labels = []
        for i, data in enumerate(test_loader):
            scores, x, xout1, xout2 = tMatchingModel(data)
            all_predicted.append(scores)
            labels = data['label'].v.float().reshape(-1).to(device)
            all_labels.append(labels)
        all_predicted = torch.cat(all_predicted)
        all_labels = torch.cat(all_labels)
        eval_loss = lossFunc(all_predicted, all_labels)
        lr_scheduler.step(eval_loss)
        all_labels = all_labels.detach().cpu().numpy().reshape((-1, FLAGS.N_TYPES))
        if FLAGS.VERBOSE:
            print(all_labels)
        all_predicted = all_predicted.detach().cpu().numpy().reshape((-1, FLAGS.N_TYPES))
        if eval_loss < best_eval_lost:
            best_eval_lost = eval_loss
            ibest = epoch
            if FLAGS.VERBOSE:
                print("New best at ", ibest, eval_loss)
            torch.save(tMatchingModel.state_dict(), "best_model.pkl")
            if eval_label is None:
                eval_label = all_labels
            best_predicted_eval_label = all_predicted
        if FLAGS.VERBOSE:
            print("Eval: ", eval_loss, all_predicted.reshape(-1), all_labels.reshape(-1), auc(all_labels, all_predicted), aupr(all_labels, all_predicted))
    print("Best eval loss, auc, aupr: ", best_eval_lost, auc(eval_label, best_predicted_eval_label),aupr(eval_label, best_predicted_eval_label), " at epoch: ", ibest)
    # print("Eval labels: ", eval_label)
    # print("Best predicted eval labels: ", best_predicted_eval_label)
if __name__ == "__main__":
    if FLAGS.RD_SEED > 0:
        print("RD SEED: ", FLAGS.RD_SEED)
        torch.manual_seed(FLAGS.RD_SEED)
        np.random.seed(FLAGS.RD_SEED)
        random.seed(FLAGS.RD_SEED)
    train()
