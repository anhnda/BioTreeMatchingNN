import torch
from SuchTree import SuchTree

import FLAGS
from model import TMatching
import pandas as pd
import seaborn
from generate_graphpair import create_graphpair


def tx():
    name = 'snow'

    T1 = SuchTree('data/plant-pollinators/' + name + '/plant.tree')
    T2 = SuchTree('data/plant-pollinators/' + name + '/animal.tree')
    links = pd.read_csv('data/plant-pollinators/' + name + '/' + name + '_links.csv', index_col=0)

    edges, edge_features, x, n1 = create_graphpair(T1, T2, links)

    tMatchingModel = TMatching(FLAGS.EDGE_FEATURE_DIM, FLAGS.EDGE_FEATURE_EMBEDDING_DIM,
                               FLAGS.NODE_FEATURE_EMBEDDING_DIM)
    x = tMatchingModel(x, edges, edge_features)
    print(x.shape, x)


if __name__ == "__main__":
    tx()
