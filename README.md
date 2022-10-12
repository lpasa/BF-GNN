# Backpropagation-free Graph Neural Networks

## Authors: Luca Pasa, Nicolò Navarin, Wolfgang Erb, Alessandro Sperduti

## Abstract

We propose a class of neural models for graphs that do not rely on backpropagation for training, thus making learning more biologically plausible and amenable to parallel implementation in hardware.
The base component of our architecture is a generalization of Gated Linear Networks which allows the adoption of multiple graph convolutions.
Specifically, each neuron is defined as a set of graph convolution filters (weight vectors) and a gating mechanism that, given a node and its topological context, selects the weight vector to use for processing the node's attributes. Two different graph processing schemes are studied, i.e.,
a message-passing aggregation scheme where the gating mechanism is embedded directly into the graph convolution, and a multi-resolution one where 
neighbouring nodes at different topological distances are jointly processed by a single graph convolution layer.
We also compare the effectiveness of different alternatives for defining the context function of a node, i.e., based on hyper-planes or on prototypes. 
A theoretical result on the expressiveness of the proposed models is also reported.
We experimented our backpropagation-free graph convolutional neural architectures on commonly adopted node classification datasets, and show competitive performances compared to the backpropagation-based counterparts.

## Cite
Accepted @ [ICDM2022](https://icdm22.cse.usf.edu/)

