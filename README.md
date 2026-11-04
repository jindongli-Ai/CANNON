# Basic Information

This code is for the submission 'No Fear of Representation Bias: Graph Contrastive Learning with Calibration and Fusion'.

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


# Implementation Details
For the Coauthor-Physics dataset, we enable an additional Gaussian random-projection step (COSTA, KDD 2022) during training (need_FA=True, FA_p=0.45). Specifically, the encoder representations are projected from \(N\) nodes to \(k=\lfloor 0.45N \rfloor\) representations before computing the contrastive objectives. This operation is used only during training for this large-scale dataset and does not modify the inference representation or the CANNON architecture.


# Reproducibility

1. We make every effort to adjust hyperparameters for baselines to achieve optimal performance. When the obtained results are similar to the reported values in the paper, we report the values in the published paper.

2. As is well known, programs are full of randomness during their operation. So we fixed the random seeds as much as possible during each run. However, the process cannot be guaranteed to be 100% identical on different machines. So when reproducing the results, it may be necessary to fine-tune the hyperparameters, but of course, it may not be necessary.


# Epoch-wise Evaluation and Validation-Based Selection

For each independent trial, CANNON generates a random 10%/10%/80% training/validation/test split once and reuses it across all evaluated outer training epochs. At a fixed evaluation interval, the learned node representations are evaluated using a logistic-regression classifier. The classifier's own checkpoint is selected by validation Micro-F1, which equals accuracy for the single-label multiclass node classification task considered here.

At the outer level, CANNON selects the earliest evaluated training epoch achieving the highest validation accuracy. It then reports the test accuracy and test Micro-F1 corresponding to that validation-selected epoch. The epoch-wise validation and test metrics are logged to make the evaluation procedure inspectable.


# Acknowledgement
This codebase is developed based on the public implementation of COSTA (KDD 2022):
https://github.com/yifeiacc/COSTA.
We sincerely thank the authors for releasing their code and facilitating reproducible research in graph contrastive learning. Parts of our training pipeline and downstream linear-evaluation implementation were adapted from the COSTA codebase.



