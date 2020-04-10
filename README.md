# Neural Symbolic Simplifier

This repo provides the code to replicate the experiments in the paper

> Hui Shi, Yang Zhang, Xinyun Chen, Yuandong Tian, Jishen Zhao <cite> Deep Symbolic Superoptimization Without Human Knowledge. in ICLR 2020 </cite>

Paper [[OpenReview](https://openreview.net/pdf?id=r1egIyBFPS)] 
A short video introduction [[YouTube](https://www.youtube.com/watch?v=SsDopKytAAg)]
## Dependency

python >= 3.5

pytorch >= 1.1.0

## Run Code

To reproduce the results, you may first download the data to the folder Data/simplify, 
as instructed [here](Data/simplify/README.md). Then run:  

> python run_optimizer.py

The hyperparameters are specified in [configs/halide.json](configs/halide.json). 
