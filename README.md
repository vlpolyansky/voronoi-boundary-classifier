# Voronoi Boundary Classifier
ICML 2019 Proceedings: [link](http://proceedings.mlr.press/v97/polianskii19a.html).

This is the initial implementation of a Voronoi Classifier - a new geometric approach for data classification. 

This repository contains two versions of the classifier, described in the paper: the main version and the approximate version. The approximate version, while better asymptotically, should only be considered for extremely large amounts of data, when a <img src="https://latex.codecogs.com/gif.latex?N^2" /> matrix cannot possibly fit into RAM memory.

## Requirements and installation

OpenCL must be installed for the main version of the algorithm. OpenMP - for an approximate version.

Additionally, `zlib` is required. In Ubuntu, that would be `sudo apt install zlib1g zlib1g-dev`.

In order to compile the program, one can execute the following commands:

```bash
mkdir build && cd build
cmake ..
make
```

Two executables will appear in `build/cpp`, namely, `VoronoiClassifier_cl` - the main algorithm, and `VoronoiClassifier_kd` - a version of the approximate algorithm, which uses a simple self-written KD-tree for an approximate nearest-neighbor search.

## Input specification

Input train (and test) data files should be given as `numpy` `.npz` files with two arrays:
 - `data` - an <img src="https://latex.codecogs.com/gif.latex?N\times%12D" /> matrix of 32-bit floats describing N D-dimensional data points
 - `labels` - an <img src="https://latex.codecogs.com/gif.latex?N\times%121" /> vector of 32-bit integers from 0 to k-1.

## Available arguments

Available program arguments for `VoronoiClassifier_cl` and `VoronoiClassifier_kd`:
 - The first argument is always a path to the first (train) dataset (npz-file).
 - If the second (test) dataset is needed, it has to be the second argument. (Remark: either this, or `--selftest is needed`)
 - `--task <classify|calc_dxdx>` The task to perform. The default task if not provided is "classify". 
 `calc_dxdx` is only available for the main version of the algorithm and computes and saves a NxN matrix needed for all further computations. This matrix can be loaded in a later use with `--dxdx`.
 - `--selftest` Initialize selftest; testing is done via "leave-one-out", test data is not required.
 - `--silent` Omit almost all output to stdout.
 - `--load <folder>` Load classification data from the given directory (to continue ray sampling from that point).
 - `--dxdx <filename>` A path to load the dxdx matrix from (generally it is faster to recompute it on a GPU).
 - `--outdir <folder>` Specify the exact output directory.
 - `--tag <string>` Specify a tag that is appended to an automatically generated output directory.
 - `--niter_a <num>` Number of "local" iterations; equal to the number of ray samplings between accuracy recalculations. Default: num=100.
 - `--niter_b <num>` Number of "global" iterations; equal to the number of accuracy recalculations. Default: num=1.
 - `--n_start <num=n_step>`, `--n_step <num=1>`, `--n_end <num=n_step*100>` Range definition for the "convergence" task.
 - `--weight <gpw|gcw|thres>` Weight function. Default if not provided: "gpw".
 - `--wsigma <num=1.0>`, `--wp <num=0.0f>`, `--wscale <num=0.0f>`, `--wthres <num=1e9>` Weight function parameters.

## See also

For a more up-to-date code related to high-dimensional Voronoi/Delaunay tesselations, please check our other repository for [Voronoi Graph Traversal](https://github.com/vlpolyansky/vgt), the modules of which can be used to easily implement or adjust the boundary classification defined here. However, at the moment, the VGT code does not support GPU computations, which is planned to be changed in the near future.

Additionally, a technique called *spoke-darts*, described in [Mitchell, Scott A., et al. "Spoke-Darts for High-Dimensional Blue-Noise Sampling"](https://arxiv.org/abs/1408.1118) uses a similar sampling methodology for Delaunay graph approximation. For more information, please check out the official VoroCrust webpage: [vorocrust.sandia.gov](https://vorocrust.sandia.gov/).
