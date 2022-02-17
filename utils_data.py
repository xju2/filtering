#!/usr/bin/env python

import torch
import numpy as np
from numpy.random import Generator, PCG64

def create_rng(seed=12345):
    rng = Generator(PCG64(seed))
    return rng

def transform(filename, do_shuffle=True):
    data = torch.load(filename, map_location='cpu')
    x = data.x.cpu().numpy()
    cell_data = data.cell_data.cpu().numpy()
    features = np.concatenate([x, cell_data], axis=1)

    edge_list = data.edge_index.cpu().numpy()

    senders = features[edge_list[0]]
    receivers = features[edge_list[1]]

    sender_pids = data.pid[edge_list[0]]
    receiver_pids = data.pid[edge_list[1]]

    inputs = np.concatenate([senders, receivers], axis=1)
    labels = (sender_pids == receiver_pids) & (sender_pids != 0)
    labels = labels.cpu().numpy().astype(int)

    if do_shuffle:
        rng = create_rng()
        p = rng.permutation(labels.shape[0])
        inputs, labels = inputs[p], labels[p]

    return inputs, labels


def read(filename):
    if type(filename) is list:
        data_list = [transform(x) for x in filename]
        inputs = np.concatenate([x[0] for x in data_list], axis=0)
        labels = np.concatenate([x[1] for x in data_list], axis=0)
    else:
        inputs, labels = transform(filename)

    return inputs, labels