# mnist

Zig implementation of a simple neural network for mnist

## Getting started

First, download the dataset from http://yann.lecun.com/exdb/mnist/. The four files you need are:

```
train-images-idx3-ubyte: training set images 
train-labels-idx1-ubyte: training set labels 
t10k-images-idx3-ubyte:  test set images 
t10k-labels-idx1-ubyte:  test set labels
```

Without renaming, put them in a `data/` folder at the root of this project's directory.

```sh
zig build
./zig-out/bin/mnist
```

## TODO

- [x] Download dataset
- [ ] SGD
