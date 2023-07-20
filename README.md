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
    Best eval loss:  tensor(0.0951, grad_fn=<MseLossBackward0>)  at epoch:  38
    Eval labels:  tensor([1., 0., 0., 1., 0., 1., 0., 0., 1., 1.])
    Best predicted eval labels:  tensor([ 1.0820,  0.6438,  0.4287,  0.6406, -0.0217,  0.8672,  0.0330,  0.0205,
         0.8839,  0.5709], grad_fn=<CatBackward0>)