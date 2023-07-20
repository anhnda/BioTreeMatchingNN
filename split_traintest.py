import json
import random

random.seed(1)
def split_jsonfile(inp, r=0.9):
    with open(inp) as f:
        js = json.load(f)
        random.shuffle(js)
        ns = int(len(js) * r)
    with open(inp[:-5] + "_train.json","w") as f:
        j1 = js[:ns]
        json.dump(j1, f)
    with open(inp[:-5] + "_test.json", "w") as f:
        j2 = js[ns:]
        json.dump(j2, f)

if __name__ == "__main__":
    split_jsonfile("positive_pairs.json")
    split_jsonfile("negative_pairs.json")
