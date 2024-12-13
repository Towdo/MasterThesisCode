# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
from scipy.io import loadmat
from PIL import Image, ImageOps
import torch
from tqdm import tqdm
from collections import Counter
from douglaspeucker import DouglasPeucker

import struct
from struct import unpack

def max_size(data):
    """larger sequence length in the data set"""
    sizes = [len(seq) for seq in data]
    return max(sizes)

def purify(strokes, max_seq_length):
    """removes to small or too long sequences + removes large gaps"""
    data = []
    for seq in strokes:
        if seq.shape[0] <= max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def calculate_normalizing_scale_factor(strokes):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

def normalize(strokes):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data

def batch(input, max_seq_length):
    output = np.zeros((max_seq_length, 5))
    length = len(input)
    output[:length, :2] = input[:length, :2]
    output[:length, 2] = 1 - input[:length, 2]
    output[:length, 3] = input[:length, 2]
    output[length:, 4] = 1

    output = torch.as_tensor(output)

    assert torch.all(torch.sum(output[:, 2:], dim = 1) == 1)
    return output



# https://github.com/pinakinathc/fscoco/blob/main/src/sbir_baseline/dataloader.py
class QuickDrawLoader(torch.utils.data.Dataset):
    def __init__(self, opt, category, mode = "train", max_seq_length = None):
        self.opt = opt

        path = os.path.join(self.opt.root_dir, 'sketchrnn', category + ".npz")
        self.vector_sketches = np.load(path, encoding='latin1', allow_pickle = True)[mode]
        if max_seq_length is None:
            max_seq_length = max([len(sketch) for sketch in self.vector_sketches])
        self.vector_sketches = purify(self.vector_sketches, max_seq_length)
        self.vector_sketches = normalize(self.vector_sketches)
        self.vector_sketches = [batch(sketch, max_seq_length) for sketch in self.vector_sketches]

        self.Nmax = max_seq_length

    def __len__(self):
        return len(self.vector_sketches)

    def __getitem__(self, index):
        return self.vector_sketches[index]



if __name__ == "__main__":
    class opt():
        root_dir = os.path.expanduser("~/data/quickdraw/")

    data = QuickDrawLoader(opt, "cat")