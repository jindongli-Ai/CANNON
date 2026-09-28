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

# Reproducibility

1. We make every effort to adjust hyperparameters for baselines to achieve optimal performance. When the obtained results are similar to the reported values in the paper, we report the values in the published paper.

2. As is well known, programs are full of randomness during their operation. So we fixed the random seeds as much as possible during each run. However, the process cannot be guaranteed to be 100% identical on different machines. So when reproducing the results, it may be necessary to fine-tune the hyperparameters, but of course, it may not be necessary.


# Evaluation Protocol and Reproducibility Note
CANNON follows the downstream linear-evaluation implementation inherited from the public COSTA codebase. Specifically, the train/validation/test split is generated inside the downstream evaluation routine. Therefore, each invocation of the evaluator independently samples a random split with the same predefined split ratio. In the GCL literature, different implementations organize random splitting differently, e.g., fixing one split for a run, evaluating final representations over multiple independently sampled splits, or re-sampling a split when the downstream evaluator is invoked. These choices concern the downstream linear-evaluation procedure rather than the self-supervised training of the GCL encoder.

A second implementation choice concerns the number of training epochs. During our reproduction of representative GCL methods, we found that reproducing the exact reported performance can be difficult when relying only on a single dataset-specific training epoch prescribed in the original implementation. Such epoch numbers are themselves experimental hyperparameters determined during model development, while the performance trajectory around the selected epoch is usually not exposed. In practice, after sufficiently long self-supervised training, GCL models generally enter a relatively stable regime, where downstream performance across neighboring epochs changes only slightly.

For this reason, CANNON performs downstream evaluation at a fixed interval, stores the corresponding results in tot_res, and prints the complete epoch-wise performance record. The results are additionally sorted by downstream performance, allowing users to directly inspect the behavior of the learned representations under different training durations instead of observing only a single pre-specified checkpoint. This design was introduced to make the sensitivity to training duration explicit and to facilitate reproduction of the reported GCL performance.

