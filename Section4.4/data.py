import os
import torch
import glob
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import json

import torch.nn.utils.rnn as rnn

from torch_geometric.data import Data, Batch


class dinoSketchSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, graphstyle = "sym_bbox"):
        self.root_dir = root_dir
        self.graphstyle = graphstyle
        self.seg_root_dir = os.path.expanduser("~/data/fscoco-seg/")

        self.ids = sorted([os.path.split(idx)[-1][:-4]
            for idx in glob.glob(os.path.join(
                root_dir, 'vector_sketches', '*.npy'))])
        
        self.segmented_ids = sorted([os.path.split(idx)[-1][:-5]
                for idx in glob.glob(os.path.join(
                    self.seg_root_dir, 'classes', '*.json'))])
        
        self.ids = [id for id in self.ids if id not in self.segmented_ids]
        
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]

        rasterized_strokes = np.load(os.path.join(self.root_dir, "rasterized_strokes_small", id + ".npy"))
        rasterized_strokes = torch.from_numpy(rasterized_strokes)
        rasterized_sketch = np.load(os.path.join(self.root_dir, "images_small", id + ".npy"))
        rasterized_sketch = torch.from_numpy(rasterized_sketch)
        if self.graphstyle == "dist" or self.graphstyle == "dist_mini":
            data = np.load(os.path.join(self.root_dir, "graphs", self.graphstyle, id + ".npy.npz"))
            spatial_edge_list = data["indices"]
            spatial_edge_list = torch.from_numpy(spatial_edge_list)
            spatial_dist = data["distances"]
            spatial_dist = torch.from_numpy(spatial_dist)
        else:
            spatial_edge_list = np.load(os.path.join(self.root_dir, "graphs", self.graphstyle, id + ".npy"))
            spatial_edge_list = torch.from_numpy(spatial_edge_list)
            spatial_dist = torch.ones(len(spatial_edge_list))
        temporal_edge_list = np.load(os.path.join(self.root_dir, "graphs", "temp", id + ".npy"))
        temporal_edge_list = torch.from_numpy(temporal_edge_list)
        
        return rasterized_sketch, rasterized_strokes, spatial_edge_list, temporal_edge_list

class dinoSketchSetAnnotated(torch.utils.data.Dataset):
    def __init__(self, root_dir, min_occurrences = None, graphstyle = "sym_bbox"):
        self.root_dir = root_dir
        self.graphstyle = graphstyle

        self.ids = sorted([os.path.split(idx)[-1][:-4]
            for idx in glob.glob(os.path.join(
                root_dir, 'vector_sketches', '*.npy'))])
        
        self.seg_root_dir = os.path.expanduser("~/data/fscoco-seg/")
        with open(self.seg_root_dir + "all_classes.json", "r") as f:
            self.all_classes = json.load(f)
        self.num_classes_ = len(self.all_classes) + 1
        self.classes_reverse_dict = {self.all_classes[i]: i for i in range(len(self.all_classes))}
        self.segmented_ids = sorted([os.path.split(idx)[-1][:-5]
                for idx in glob.glob(os.path.join(
                    self.seg_root_dir, 'classes', '*.json'))])
        
        self.ids = [id for id in self.ids if id in self.segmented_ids]

        if min_occurrences is not None:
            from collections import Counter

            c = Counter()

            for id in self.ids:
                with open(self.seg_root_dir + "/classes/" + str(id) + ".json") as f:
                    classes = json.load(f)
                    c.update(set(classes))

            self.all_classes = [k for k in self.all_classes if c[k] > min_occurrences]
            print(self.all_classes)
            self.num_classes_ = len(self.all_classes) + 1
            self.classes_reverse_dict = {self.all_classes[i]: i for i in range(len(self.all_classes))}
        
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]

        rasterized_strokes = np.load(os.path.join(self.root_dir, "rasterized_strokes_small", id + ".npy"))
        rasterized_strokes = torch.from_numpy(rasterized_strokes)
        rasterized_sketch = np.load(os.path.join(self.root_dir, "images_small", id + ".npy"))
        rasterized_sketch = torch.from_numpy(rasterized_sketch)
        if self.graphstyle == "dist" or self.graphstyle == "dist_mini":
            data = np.load(os.path.join(self.root_dir, "graphs", self.graphstyle, id + ".npy.npz"))
            spatial_edge_list = data["indices"]
            spatial_edge_list = torch.from_numpy(spatial_edge_list)
            spatial_dist = data["distances"]
            spatial_dist = torch.from_numpy(spatial_dist)
        else:
            spatial_edge_list = np.load(os.path.join(self.root_dir, "graphs", self.graphstyle, id + ".npy"))
            spatial_edge_list = torch.from_numpy(spatial_edge_list)
            spatial_dist = torch.zeros(len(spatial_edge_list))
        temporal_edge_list = np.load(os.path.join(self.root_dir, "graphs", "temp", id + ".npy"))
        temporal_edge_list = torch.from_numpy(temporal_edge_list)

        with open(os.path.join(self.seg_root_dir, 'classes', id + '.json'), "r") as f:
            labels = json.load(f)
        labels = torch.tensor(list(map(lambda x: self.classes_reverse_dict.get(x, -1) + 1, labels)))
        return rasterized_sketch, rasterized_strokes, spatial_edge_list, temporal_edge_list, labels
    
# class dinoSketchSetAnnotatedPlusVector(torch.utils.data.Dataset):
#     def __init__(self, root_dir, min_occurrences = None, graphstyle = "sym_bbox"):
#         self.root_dir = root_dir
#         self.graphstyle = graphstyle

#         self.ids = sorted([os.path.split(idx)[-1][:-4]
#             for idx in glob.glob(os.path.join(
#                 root_dir, 'vector_sketches', '*.npy'))])
        
#         self.seg_root_dir = os.path.expanduser("~/data/fscoco-seg/")
#         with open(self.seg_root_dir + "all_classes.json", "r") as f:
#             self.all_classes = json.load(f)
#         self.num_classes_ = len(self.all_classes) + 1
#         self.classes_reverse_dict = {self.all_classes[i]: i for i in range(len(self.all_classes))}
#         self.segmented_ids = sorted([os.path.split(idx)[-1][:-5]
#                 for idx in glob.glob(os.path.join(
#                     self.seg_root_dir, 'classes', '*.json'))])
        
#         self.ids = [id for id in self.ids if id in self.segmented_ids]

#         if min_occurrences is not None:
#             from collections import Counter

#             c = Counter()

#             for id in self.ids:
#                 with open(self.seg_root_dir + "/classes/" + str(id) + ".json") as f:
#                     classes = json.load(f)
#                     c.update(set(classes))

#             self.all_classes = [k for k in self.all_classes if c[k] > min_occurrences]

#             self.num_classes_ = len(self.all_classes) + 1
#             self.classes_reverse_dict = {self.all_classes[i]: i for i in range(len(self.all_classes))}
        
        
#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, index):
#         id = self.ids[index]

#         rasterized_strokes = np.load(os.path.join(self.root_dir, "rasterized_strokes_small", id + ".npy"))
#         rasterized_strokes = torch.from_numpy(rasterized_strokes)
#         rasterized_sketch = np.load(os.path.join(self.root_dir, "images_small", id + ".npy"))
#         rasterized_sketch = torch.from_numpy(rasterized_sketch)
#         if self.graphstyle == "dist" or self.graphstyle == "dist_mini":
#             data = np.load(os.path.join(self.root_dir, "graphs", self.graphstyle, id + ".npy.npz"))
#             spatial_edge_list = data["indices"]
#             spatial_edge_list = torch.from_numpy(spatial_edge_list)
#             spatial_dist = data["distances"]
#             spatial_dist = torch.from_numpy(spatial_dist)
#         else:
#             spatial_edge_list = np.load(os.path.join(self.root_dir, "graphs", self.graphstyle, id + ".npy"))
#             spatial_edge_list = torch.from_numpy(spatial_edge_list)
#             spatial_dist = torch.zeros(len(spatial_edge_list))
#         temporal_edge_list = np.load(os.path.join(self.root_dir, "graphs", "temp", id + ".npy"))
#         temporal_edge_list = torch.from_numpy(temporal_edge_list)

#         vector_sketch = np.load(os.path.join(self.root_dir, "vector_sketches", id + ".npy"))
#         vector_sketch = torch.from_numpy(vector_sketch)

#         with open(os.path.join(self.seg_root_dir, 'classes', id + '.json'), "r") as f:
#             labels = json.load(f)
#         labels = torch.tensor(list(map(lambda x: self.classes_reverse_dict.get(x, -1) + 1, labels)))
#         return vector_sketch, rasterized_sketch, rasterized_strokes, spatial_edge_list, temporal_edge_list, spatial_dist, labels
        

# class dinoSketchSetAnnotated(torch.utils.data.Dataset):
#     def __init__(self, root_dir, padding = None):
#         self.root_dir = root_dir
#         self.padding = padding

#         self.ids = sorted([os.path.split(idx)[-1][:-4]
#             for idx in glob.glob(os.path.join(
#                 root_dir, 'vector_sketches', '*.npy'))])
        
#         self.seg_root_dir = os.path.expanduser("~/data/fscoco-seg/")
#         with open(self.seg_root_dir + "all_classes.json", "r") as f:
#             self.all_classes = json.load(f)
#         self.num_classes_ = len(self.all_classes)
#         self.classes_reverse_dict = {self.all_classes[i]: i for i in range(len(self.all_classes))}
#         self.segmented_ids = sorted([os.path.split(idx)[-1][:-5]
#                 for idx in glob.glob(os.path.join(
#                     self.seg_root_dir, 'classes', '*.json'))])
        
#         self.ids = [id for id in self.ids if id in self.segmented_ids]
        
        
#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, index):
#         id = self.ids[index]

#         vector_sketch = np.load(os.path.join(self.root_dir, "vector_sketches", id + ".npy"))
#         vector_sketch = torch.from_numpy(vector_sketch)
#         rasterized_sketch = Image.open(os.path.join(self.root_dir, "raster_sketches", id + ".jpg"))
#         rasterized_sketch = TF.pil_to_tensor(rasterized_sketch)

#         with open(os.path.join(self.seg_root_dir, 'classes', id + '.json'), "r") as f:
#             labels = json.load(f)
#         labels = labels
#         # labels = torch.tensor(list(map(lambda x: self.classes_reverse_dict.get(x, -1), labels)))
#         labels = torch.tensor(list(map(lambda x: self.classes_reverse_dict.get(x, -1), labels)))
#         return vector_sketch, None, rasterized_sketch, labels
    