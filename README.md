# BioTreeMatchingNN

# Setup

```shell
    conda env create -f envx.yml
    conda activate btm
```
## Data:

    Download the data folder at https://github.com/ryneches/SuchTree/ and copy to current folder

## Test run:

    python3 training.py

### Current run:
    Best eval loss, auc, aupr:  tensor(0.1497, grad_fn=<MseLossBackward0>) 0.9600000000000001 0.9666666666666666  at epoch:  28
    Eval labels:  [0. 0. 1. 1. 0. 0. 1. 1. 0. 1.]
    Best predicted eval labels:  [ 0.9016529  -0.00399305  0.9675149   0.86232644  0.7967971   0.00653598
    0.90543926  0.98692226  0.14010811  0.9804735 ]