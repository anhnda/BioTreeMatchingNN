import random
from abc import ABC

from torch_geometric.data import Dataset

import FLAGS
from data_factory.generate_graphpair import create_graphpair
import json
from SuchTree import SuchTree
import pandas as pd
from utils.func import get_insert_dict_index_with_offset_id

class BTDataset(Dataset, ABC):
    def __init__(self, pos_path="positive_pairs.json", neg_path="negative_pairs.json", tp_dict = None):
        super(BTDataset, self).__init__()
        with open(pos_path) as f:
            self.positive_files = json.load(f)
            print("Pos len: ", len(self.positive_files))
        with open(neg_path) as f:
            self.negative_files = json.load(f)
            print("Neg len: ", len(self.negative_files))
        self.graphs = None
        self.tp_dict = tp_dict
        self.load_data()

    def len(self):
        return len(self.positive_files) + len(self.negative_files)

    def load_data(self):
        graphs = []
        print("Loading...")
        max_depth = -10
        tps = self.tp_dict

        for lb, samples in enumerate([self.positive_files, self.negative_files]):
            # print("LB: ", lb)
            for sample in samples:
                # print(sample['host'], sample['guest'], sample['links'])
                tHost = SuchTree(sample['host'])
                tGuest = SuchTree(sample['guest'])
                if FLAGS.BINARY_LABEL:
                    lbx = lb
                else:
                    lbx = get_insert_dict_index_with_offset_id(tps, sample['type'])
                max_depth = max(max_depth, tHost.depth, tGuest.depth)
                links = pd.read_csv(sample['links'], index_col=0)
                graph = create_graphpair(tHost, tGuest, links, lbx,nn=sample['links'])
                graphs.append(graph)
        random.shuffle(graphs)
        self.graphs = graphs
        print("Loaded!")
        print("Max depth: ", max_depth)
        print("Tps: ", len(tps), tps)

    def get(self, idx: int):
        return self.graphs[idx]


if __name__ == "__main__":
    tp_dict = {}
    train_dataset = BTDataset(pos_path="../jsinfo/positive_pairs_train.json", neg_path="../jsinfo/negative_pairs_train.json", tp_dict=tp_dict)
    print(len(train_dataset))
    print(tp_dict)
