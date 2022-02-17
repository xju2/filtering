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

def read_and_save(in_fname, out_fname):    
    if os.path.exists(out_fname+".npz"):
        return 

    inputs, labels = read(in_fname)
    np.savez(out_fname, inputs, labels)

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Convert files to a matrix')
    add_arg = parser.add_argument
    add_arg('indir', help='input directory')
    add_arg('outdir', help='output directory')
    add_arg("-w", "--workers", help='number of workers', type=int, default=1)
    args = parser.parse_args()
    
    indir,outdir, workers = args.indir, args.outdir, args.workers
    input_list = []
    for root,dirs,files in os.walk(args.indir):

        for fname in files:
            if "pdf" in fname: continue
            else:
                infname = os.path.join(root, fname)
                outfname = os.path.join(root.replace(indir, outdir), fname)
                os.makedirs(os.path.dirname(outfname), exist_ok=True)
                input_list.append( (infname, outfname) )

    print("total input files", len(input_list), f"using {args.workers} workers")

    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as p:
            p.starmap(read_and_save, input_list)
    else:
        for infname,outfname in input_list:
            read_and_save(infname, outfname)