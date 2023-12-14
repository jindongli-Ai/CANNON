# Basic Information

This code is for the submission for Pattern Recognition 2024 ('No Fear of Representation Bias: Graph Contrastive Learning with Calibration and Fusion').

The paper is under reivew. Please don't quote or use it for other purposes at present.


# Requirements

This code requires the following:

- Python==3.8
- Pytorch==1.11.0
- Pytorch Geometric==2.0.4
- PyGCL==0.1.2
- GCL==1.1.0.cu113
- Numpy==1.23.1
- OGB==1.3.3

# Dataset
We use public graph benchmark. 

When running the code, they will be automatically downloaded from the corresponding library. No manual download required.


# Hardware used for inplementation

- GPU: A40 (48GB)
- OS: Linux (Slurm)

# Usage

Run node classification on CiteSeer dataset:
```
bash script/citeseer.sh
```

# Reproducibility

1. We make every effort to adjust hyperparameters for baselines to achieve optimal performance. When the obtained results are similar to the reported values in the paper, we report the values in the published paper.

2. As is well known, programs are full of randomness during their operation. So we fixed the random seeds as much as possible during each run. However, the process cannot be guaranteed to be 100% identical on different machines. So when reproducing the results, it may be necessary to fine-tune the hyperparameters, but of course, it may not be necessary.

