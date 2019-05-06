# Voronoi Classifier
ICML 2019 Proceedings: [[link]]

This is the initial implementation of a Voronoi Classifier - a new geometric approach for data classification. 

This repository contains two versions of the classifier, described in the paper: the main version and the approximate version. The approximate version, while better asymptotically, should only be considered for extremely large amounts of data, when a <img src="https://latex.codecogs.com/gif.latex?N^2" /> matrix cannot possibly fit into RAM memory.

## Requirements and installation

TBD

OpenCL must be installed for the main version of the algorithm. OpenMP - for an approximate version.

Additionally, `zlib` is required. In Ubuntu, that would be `sudo apt install zlib1g`.

In order to compile the program, one can execute the following commands:

```bash
mkdir build && cd build
cmake ..
make
```

Two executables will appear in `build/cpp`, namely, `VoronoiClassifier_cl` - the main algorithm, and `VoronoiClassifier_kd` - a version of the approximate algorithm, which uses a simple self-written KD-tree for an approximate nearest-neighbor search.

## Input and output specification

Input train (and test) data files should be given as `numpy` `.npz` files with two arrays:
 - `data` - an <img src="https://latex.codecogs.com/gif.latex?N\times%12D" /> matrix of 32-bit floats describing N D-dimensional data points
 - `labels` - an <img src="https://latex.codecogs.com/gif.latex?N\times%121" /> vector of 32-bit integers from 0 to k-1.

## Available arguments

