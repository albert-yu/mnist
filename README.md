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

Without renaming the files, put them in a `data/` folder at the root of this project's directory.

```sh
zig build
./zig-out/bin/mnist
```

On my machine, getting around 93% accuracy with 100 epochs.

## Motivation

This project was a way for me to (1) learn the fundamentals of neural networks and (2) learn [Zig](https://github.com/ziglang/zig).

I started with [Michael Nielsen's excellent introduction to neural networks](http://neuralnetworksanddeeplearning.com/index.html) and the Python implementation within as a reference. But translating the Python code directly to Zig was a clumsy effort. By chance, I stumbled upon [this blog post implementing mnist in Zig](https://monadmonkey.com/dnns-from-scratch-in-zig). Taking inspiration from there, I refactored my original implementation to hard-code the layers and feedforward/backprop them explicitly instead of using a loop.

## Room for improvement

Right now, it's a very basic, slow implementation. Ideas for improvement:

1. Leverage a BLAS library or similar for faster matrix multiplication
2. Batch inputs and outputs in a concatenated matrix rather than loop through each data point individually (would benefit from implementing (1) first)
3. ~Use arena allocation for less memory allocation overhead~ performance didn't improve as much as I'd hoped
