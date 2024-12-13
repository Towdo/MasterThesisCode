#!/usr/bin/env python
# coding: utf-8

# In[20]:
import sys

import torch

import os

from data import dinoSketchSet, dinoSketchSetAnnotated

from transformers import CLIPImageProcessor, CLIPVisionModel
import matplotlib.pyplot as plt

import torch.nn.utils.rnn as rnn
import torch.nn as nn
import torch.nn.functional as F

import torchvision.ops as ops
import torchvision.transforms as T

import scipy

from tqdm import tqdm
from torch_geometric.data import HeteroData, Batch

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint

import tempfile

from sklearn.metrics import accuracy_score


# In[21]:


upscale = 1
device = "cuda"


# In[22]:


def collate_fn(data):
        rasterized_sketch, rasterized_strokes, spatial_edge_list, temporal_edge_list, spatial_dist, labels = zip(*data)
        return torch.utils.data.default_collate(rasterized_sketch), rnn.pack_sequence(rasterized_strokes, enforce_sorted=False), rnn.pack_sequence(spatial_edge_list, enforce_sorted=False), rnn.pack_sequence(temporal_edge_list, enforce_sorted=False), rnn.pack_sequence(spatial_dist, enforce_sorted=False), rnn.pack_sequence(labels, enforce_sorted=False)


# In[23]:


node_types = ["stroke"]
edge_types = [("stroke", "spatial", "stroke"),
            ("stroke", "temporal", "stroke")]
metadata = (node_types, edge_types)

from torch_geometric.nn import HGTConv, Linear, HeteroConv, GATConv, GraphConv, Sequential

# class HGT(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
#         super().__init__()

#         self.lin_dict = torch.nn.ModuleDict()
#         for node_type in node_types:
#             self.lin_dict[node_type] = Linear(-1, hidden_channels)

#         self.convs = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             conv = HGTConv(hidden_channels, hidden_channels, metadata,
#                            num_heads)#, group='sum')
#             self.convs.append(conv)

#         self.lin = Linear(hidden_channels, out_channels)

#     def forward(self, x_dict, edge_index_dict):
#         for node_type, x in x_dict.items():
#             x_dict[node_type] = self.lin_dict[node_type](x).relu_()

#         # print(x_dict, edge_index_dict)
#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)

#         return self.lin(x_dict['stroke'])

class LinearHGT(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        # print(x_dict)
        return {"stroke": self.linear(x_dict["stroke"])}
    
class AttentionHGT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim

        self.attention = nn.TransformerEncoderLayer(in_dim, 1, dim_feedforward = hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        # print(x_dict)
        return {"stroke": self.linear(self.attention(x_dict["stroke"]))}

class AttentionMiniHGT(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.attention = nn.TransformerEncoderLayer(in_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        # print(x_dict)
        return {"stroke": self.attention(x_dict["stroke"])}


# In[24]:


def train_f(config):
    print(config)
    clip_level_local = config["clip_level"]

    if config["vit"] == "clip":
        clip = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch16",
        device_map=device,
        )
        num_patches = 14
        feature_dim = 768
    elif config["vit"] == "dino":
        # Setup DINO with hooks
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                # Store the patch tokens, excluding CLS token
                activations[name] = output[:, 1:, :]
            return hook

        num_patches = 16
        feature_dim = 384

        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device = device)

        # Register hooks for each block
        for i, block in enumerate(dino.blocks):
            block.register_forward_hook(get_activation(f'block_{i}'))

    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    root = os.path.expanduser("~/data/fscoco-clone/")
    dataset = dinoSketchSetAnnotated(root, min_occurrences=None if config["hardmode"] else 20, graphstyle = config["graphstyle"]) #graphstyle
    print(dataset.num_classes_)

    generator = torch.Generator().manual_seed(42)   # We seed the random_split to be able to reproduce the same train / test split
    trainset, testset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size = config["batch_size"], shuffle = True, collate_fn = collate_fn)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = config["batch_size"], shuffle = True, collate_fn = collate_fn)

    if config["model_type"] == "Linear":
        model = LinearHGT(feature_dim, dataset.num_classes_).to(device = device)
    elif config["model_type"] == "HeteroGraphConv":
        model = Sequential("x_dict, edge_index_dict, edge_weight_dict",
            [(HeteroConv({
            ("stroke", "spatial", "stroke"): GraphConv(-1, feature_dim),
            ("stroke", "temporal", "stroke"): GraphConv(-1, feature_dim)
            }), "x_dict, edge_index_dict, edge_weight_dict -> x_dict") for _ in range(config["num_layers"] - 1)]
            + [(HeteroConv({
            ("stroke", "spatial", "stroke"): GraphConv(-1, dataset.num_classes_),
            ("stroke", "temporal", "stroke"): GraphConv(-1, dataset.num_classes_)
            }), "x_dict, edge_index_dict, edge_weight_dict -> x_dict")]
        ).to(device = device)
    elif config["model_type"] == "HeteroGAT":
        model = Sequential("x_dict, edge_index_dict",
            [(HeteroConv({
            ("stroke", "spatial", "stroke"): GATConv(-1, feature_dim),
            ("stroke", "temporal", "stroke"): GATConv(-1, feature_dim)
            }), "x_dict, edge_index_dict -> x_dict") for _ in range(config["num_layers"] - 1)]
            + [(HeteroConv({
            ("stroke", "spatial", "stroke"): GATConv(-1, dataset.num_classes_),
            ("stroke", "temporal", "stroke"): GATConv(-1, dataset.num_classes_)
            }), "x_dict, edge_index_dict -> x_dict")]
        ).to(device = device)
    elif config["model_type"] == "Attention":
        model = AttentionHGT(feature_dim, dataset.num_classes_).to(device = device)
    elif config["model_type"] == "AttentionMini":
        model = AttentionMiniHGT(feature_dim, dataset.num_classes_).to(device = device)
    else:
        raise Exception()

    optim = torch.optim.Adam(model.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])

    def get_bbox(stroke):
        result = torch.cat([torch.min(stroke[:, :2], 0)[0], torch.max(stroke[:, :2], 0)[0]], 0)
        assert result[0] <= result[2] and result[1] <= result[3]
        return result

    for epoch in tqdm(range(32), disable = True):
        train_loss = 0
        for dat in tqdm(train_loader, disable = True):
            rasterized_sketch, rasterized_strokes, spatial_edge_list, temporal_edge_list, spatial_dist, labels = dat
            optim.zero_grad()
            rasterized_sketch = rasterized_sketch.to(device = device)
            rasterized_strokes = rasterized_strokes.to(device = device)
            spatial_edge_list = spatial_edge_list.to(device = device)
            temporal_edge_list = temporal_edge_list.to(device = device)
            spatial_dist = spatial_dist.to(device = device)
            labels = labels.to(device = device)
            images = processor(rasterized_sketch.unsqueeze(1).expand(-1, 3, -1, -1), return_tensors="pt", do_rescale=False)
            
            if config["vit"] == "clip":
                features = clip(pixel_values = images["pixel_values"].to(device = device), output_hidden_states = True).hidden_states
                patch_features = features[clip_level_local][:, 1:].reshape(-1, num_patches, num_patches, feature_dim)
            elif config["vit"] == "dino":
                final_feature = dino.forward_features(images["pixel_values"].to(device=device))["x_norm_patchtokens"]

                # Stack features like CLIP format
                features = torch.stack([activations[f'block_{i}'] for i in range(len(dino.blocks))] + [final_feature], 0)
                patch_features = features[clip_level_local, :].reshape(-1, num_patches, num_patches, feature_dim)
            else:
                raise Exception()
            

            unpacked_rasterized_strokes = rnn.unpack_sequence(rasterized_strokes)
            unpacked_spatial_edges = rnn.unpack_sequence(spatial_edge_list)
            unpacked_temporal_edges = rnn.unpack_sequence(temporal_edge_list)
            unpacked_spatial_distances = rnn.unpack_sequence(spatial_dist)
            unpacked_labels = rnn.unpack_sequence(labels)
            graphs = []
            for i in range(len(rasterized_sketch)):
                stroke_pixel_sum = [(~s).reshape(num_patches, 224 // num_patches, num_patches, 224 // num_patches).sum(dim=(1, 3)) for s in unpacked_rasterized_strokes[i]]
                features = torch.stack([(patch_features[i] * s[..., None]).sum(dim=(0, 1)) / s.sum().clamp(1) for s in stroke_pixel_sum])

                graph = HeteroData()
                graph["stroke"].x = features
                graph["stroke", "spatial", "stroke"].edge_indices = unpacked_spatial_edges[i].to(device = device)
                graph["stroke", "temporal", "stroke"].edge_indices = unpacked_temporal_edges[i].to(device = device)
                if config["graphstyle"] == "dist" or config["graphstyle"] == "dist_mini":
                    dist = torch.exp(- unpacked_spatial_distances[i]**2 / config["sigma"] ** 2)
                    graph["stroke", "spatial", "stroke"].edge_weight = dist
                else:
                    graph["stroke", "spatial", "stroke"].edge_weight = torch.ones(len(graph["stroke", "spatial", "stroke"].edge_indices), device = device)
                graph["stroke"].label = torch.cat([unpacked_labels[i], unpacked_labels[i][[-1],]])[:len(unpacked_rasterized_strokes[i])]
                graph["stroke"].logits = 0

                assert len(graph["stroke"].label) == len(graph["stroke"].x)
                
                graphs.append(graph)
            batch = Batch.from_data_list(graphs)
            
            #print(model(batch.x_dict, {k: batch.edge_indices_dict[k].T for k in batch.edge_indices_dict}))
            if config["model_type"] == "HeteroGraphConv":
                batch["stroke"].logits = model(batch.x_dict, {k: batch.edge_indices_dict[k].T for k in batch.edge_indices_dict}, edge_weight_dict = {('stroke', 'spatial', 'stroke'): batch.edge_weight_dict["stroke", "spatial", "stroke"]})["stroke"]
            else:
                batch["stroke"].logits = model(batch.x_dict, {k: batch.edge_indices_dict[k].T for k in batch.edge_indices_dict})["stroke"]
            
            # print(torch.min(batch["stroke"].label), torch.max(batch["stroke"].label), batch["stroke"].logits.shape)
            mask = batch["stroke"].label != 0
            if torch.count_nonzero(mask) == 0:
                continue
            # print(torch.count_nonzero(batch["stroke"].label == -1))
            loss = F.cross_entropy(batch["stroke"].logits[mask], batch["stroke"].label[mask])

            assert not loss.isnan() and not loss.isinf()

            loss.backward()
            optim.step()
            train_loss += loss.item()
        # print("Train loss", train_loss / len(train_loader))

        test_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for dat in tqdm(test_loader, disable = True):
                rasterized_sketch, rasterized_strokes, spatial_edge_list, temporal_edge_list, spatial_dist, labels = dat
                rasterized_sketch = rasterized_sketch.to(device = device)
                rasterized_strokes = rasterized_strokes.to(device = device)
                spatial_edge_list = spatial_edge_list.to(device = device)
                temporal_edge_list = temporal_edge_list.to(device = device)
                spatial_dist = spatial_dist.to(device = device)
                labels = labels.to(device = device)
                images = processor(rasterized_sketch.unsqueeze(1).expand(-1, 3, -1, -1), return_tensors="pt", do_rescale=False)
                
                if config["vit"] == "clip":
                    features = clip(pixel_values = images["pixel_values"].to(device = device), output_hidden_states = True).hidden_states
                    patch_features = features[clip_level_local][:, 1:].reshape(-1, num_patches, num_patches, feature_dim)
                elif config["vit"] == "dino":
                    final_feature = dino.forward_features(images["pixel_values"].to(device=device))["x_norm_patchtokens"]

                    # Stack features like CLIP format
                    features = torch.stack([activations[f'block_{i}'] for i in range(len(dino.blocks))] + [final_feature], 0)
                    patch_features = features[clip_level_local, :].reshape(-1, num_patches, num_patches, feature_dim)
                else:
                    raise Exception()

                unpacked_rasterized_strokes = rnn.unpack_sequence(rasterized_strokes)
                unpacked_spatial_edges = rnn.unpack_sequence(spatial_edge_list)
                unpacked_temporal_edges = rnn.unpack_sequence(temporal_edge_list)
                unpacked_spatial_distances = rnn.unpack_sequence(spatial_dist)
                unpacked_labels = rnn.unpack_sequence(labels)
                graphs = []
                for i in range(len(rasterized_sketch)):
                    stroke_pixel_sum = [(~s).reshape(num_patches, 224 // num_patches, num_patches, 224 // num_patches).sum(dim=(1, 3)) for s in unpacked_rasterized_strokes[i]]
                    features = torch.stack([(patch_features[i] * s[..., None]).sum(dim=(0, 1)) / s.sum().clamp(1) for s in stroke_pixel_sum])

                    graph = HeteroData()
                    graph["stroke"].x = features
                    graph["stroke", "spatial", "stroke"].edge_indices = unpacked_spatial_edges[i].to(device = device)
                    graph["stroke", "temporal", "stroke"].edge_indices = unpacked_temporal_edges[i].to(device = device)
                    if config["graphstyle"] == "dist" or config["graphstyle"] == "dist_mini":
                        dist = torch.exp(- unpacked_spatial_distances[i]**2 / config["sigma"] ** 2)
                        graph["stroke", "spatial", "stroke"].edge_weight = dist
                    else:
                        graph["stroke", "spatial", "stroke"].edge_weight = torch.ones(len(graph["stroke", "spatial", "stroke"].edge_indices), device = device)
                    graph["stroke"].label = torch.cat([unpacked_labels[i], unpacked_labels[i][[-1],]])[:len(unpacked_rasterized_strokes[i])]
                    graph["stroke"].logits = 0

                    assert len(graph["stroke"].label) == len(graph["stroke"].x)
                    
                    graphs.append(graph)
                batch = Batch.from_data_list(graphs)
                
                #print(model(batch.x_dict, {k: batch.edge_indices_dict[k].T for k in batch.edge_indices_dict}))
                if config["model_type"] == "HeteroGraphConv":
                    batch["stroke"].logits = model(batch.x_dict, {k: batch.edge_indices_dict[k].T for k in batch.edge_indices_dict}, edge_weight_dict = {('stroke', 'spatial', 'stroke'): batch.edge_weight_dict["stroke", "spatial", "stroke"]})["stroke"]
                else:
                    batch["stroke"].logits = model(batch.x_dict, {k: batch.edge_indices_dict[k].T for k in batch.edge_indices_dict})["stroke"]
                
                # print(torch.min(batch["stroke"].label), torch.max(batch["stroke"].label), batch["stroke"].logits.shape)
                mask = batch["stroke"].label != 0
                if torch.count_nonzero(mask) == 0:
                    continue

                loss = F.cross_entropy(batch["stroke"].logits[mask], batch["stroke"].label[mask])
                accuracy = (torch.argmax(batch["stroke"].logits[mask], -1) == batch["stroke"].label[mask]).float().mean()

                test_loss += loss.item()
                
                preds = torch.argmax(batch["stroke"].logits[mask], -1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["stroke"].label[mask].cpu().numpy())
            # print("Test loss", test_loss / len(test_loader))
        
        test_accuracy = accuracy_score(all_labels, all_preds)

        train.report(
            {"train_loss": train_loss / len(train_loader), "test_loss": test_loss / len(test_loader), "test_acc": test_accuracy},
        )
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        # Save model state
        checkpoint_path = os.path.join(temp_checkpoint_dir, "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'config': config,
            'feature_dim': feature_dim,
        }, checkpoint_path)
        
        # Create final checkpoint for Ray
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        train.report(
            {
                "train_loss": train_loss / len(train_loader),
                "test_loss": test_loss / len(test_loader),
                "test_acc": test_accuracy,
            },
            checkpoint=checkpoint
        )


# In[25]:


# config = {
#     "batch_size": 10,#tune.grid_search([1, 5, 10, 20]),#tune.grid_search([20, 40, 60, 80, 20, 30, 40, 50]),
#     "model_type": sys.argv[1],#"HeteroConv", #"Linear"
#     # "hidden_size": tune.grid_search([4, 16, 128]),
#     # "num_heads": tune.grid_search([1, 4]),
#     # "num_layers": tune.grid_search([1, 4]),
#     "clip_level": int(sys.argv[2]),#tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
#     "lr": tune.grid_search([0.005, 0.001, 0.0005, 0.0001]),
#     "weight_decay": tune.grid_search([0.01, 0.001, 0.0001]),
# }

config = {
    "batch_size": 5 if sys.argv[2] != "Attention" and sys.argv[2] != "AttentionMini" else 1,#tune.grid_search([1, 5, 10, 20]),#tune.grid_search([20, 40, 60, 80, 20, 30, 40, 50]),
    "graphstyle": sys.argv[4],#tune.grid_search(["sym_bbox", "asym_bbox", "dist"]),
    "vit": sys.argv[1],#tune.grid_search(["clip", "dino"]),#"clip","dino"
    "model_type": sys.argv[2],#"Linear", "HeteroGraphConv", "HeteroGAT"
    "clip_level": int(sys.argv[3]),#tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    "lr": 0.0001,
    "weight_decay": 0.001,
    "sigma": 4 if sys.argv[4] == "dist" else 1,
    "num_layers": tune.grid_search([1, 2, 3, 4]),
    "hardmode": True,
}

from ray.tune.stopper import Stopper
from typing import Dict

class EarlyStopper(Stopper):
    def __init__(self, patience: int = 10, metric = "test_loss"):
        self.losses = {}
        self.patience = patience
        self.metric = metric

    def __call__(self, trial_id, result):
        self.losses[trial_id] = self.losses.get(trial_id, []) + [result[self.metric]]

        if len(self.losses[trial_id]) < self.patience:
            return False
        return min(self.losses[trial_id][-self.patience:]) > min(self.losses[trial_id])

    def stop_all(self):
        return False

# scheduler = tune.schedulers.ASHAScheduler(
#         metric="test_loss",
#         mode="min",
#         max_t=50,
#         grace_period=5,
#         reduction_factor=2)

lockfile_path = "experiment.lock"
csv_file_path = "final_results.csv"

import pandas as pd
from filelock import FileLock

analysis = tune.run(
    train_f,
    resources_per_trial={"cpu": 8, "gpu": 0.25} if sys.argv[2] != "Attention" else {"cpu": 16, "gpu": 0.5},
    config=config,
    num_samples=1,
    # stop=EarlyStopper(patience = 5, metric = "test_loss"),
    # scheduler = scheduler
)

df = analysis.dataframe()

with FileLock(lockfile_path):  # Ensure only one process writes at a time
    # First, initialize the CSV if not already done
    if not os.path.exists(csv_file_path):
        df.to_csv(csv_file_path, mode='w', header=True, index=False)
    else: # Append results to the CSV
        df.to_csv(csv_file_path, mode='a', header=False, index=False)
    print(f"Final analysis results saved to {csv_file_path}")

