import sys

import torch

import os

from data import dinoSketchSet, dinoSketchSetAnnotatedPlusVector
from rasterize import rasterize_together, rasterize

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

import glob

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

if __name__ == "__main__":
    print("Should only be printed once")

    upscale = 1
    device = "cuda"
    mode = "hard"
     
    batch_size = 5


    def collate_fn(data):
            rasterized_sketch, rasterized_strokes, spatial_edge_list, temporal_edge_list, spatial_dist, labels = zip(*data)
            return torch.utils.data.default_collate(rasterized_sketch), rnn.pack_sequence(rasterized_strokes, enforce_sorted=False), rnn.pack_sequence(spatial_edge_list, enforce_sorted=False), rnn.pack_sequence(temporal_edge_list, enforce_sorted=False), rnn.pack_sequence(spatial_dist, enforce_sorted=False), rnn.pack_sequence(labels, enforce_sorted=False)



    node_types = ["stroke"]
    edge_types = [("stroke", "spatial", "stroke"),
                ("stroke", "temporal", "stroke")]
    metadata = (node_types, edge_types)

    from torch_geometric.nn import HGTConv, Linear, HeteroConv, GATConv, GraphConv, Sequential

    def find_checkpoint(experiment_name):
        ray_results = os.path.expanduser("~/ray_results")
        matches = glob.glob(os.path.join(ray_results, "**", experiment_name + "*", "checkpoint_000000/model.pt"), recursive=True)
        if not matches:
            raise ValueError(f"No checkpoint found for experiment: {experiment_name}")
        return matches[0]

    # Load your saved model checkpoint
    checkpoint_path = find_checkpoint("train_f_e2845_00001")  # Update this
    #checkpoint_path = find_checkpoint("train_f_8d627_00001")
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']

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

            self.attention = nn.TransformerEncoderLayer(in_dim, hidden_dim)
            self.linear = nn.Linear(hidden_dim, out_dim)

        def forward(self, x_dict, edge_index_dict):
            # print(x_dict)
            return {"stroke": self.linear(self.attention(x_dict["stroke"]))}


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
    dataset = dinoSketchSetAnnotatedPlusVector(root, min_occurrences=None if mode=="hard" else 20, graphstyle = config["graphstyle"]) #graphstyle
    print(dataset.num_classes_)

    generator = torch.Generator().manual_seed(42)   # We seed the random_split to be able to reproduce the same train / test split
    _, testset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator)

    # train_loader = torch.utils.data.DataLoader(trainset, batch_size = config["batch_size"], shuffle = True, collate_fn = collate_fn)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size = config["batch_size"], shuffle = False, collate_fn = collate_fn)

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
    else:
        raise Exception()

    print(next(model.parameters()))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(next(model.parameters()))
    # raise Exception("Hi")
    model.eval()

    # Load dataset
    # generator = torch.Generator().manual_seed(42)
    # _, testset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        dat = testset[77]#[17]  #0 sheep, 35 giraffe   , 48 dog, 49 patch_limn, 54 hydrant, 59 signboard, 68 weird dog
        vector_sketch, rasterized_sketch, rasterized_strokes, spatial_edge_list, temporal_edge_list, spatial_dist, labels = dat
        rasterized_sketch = rasterized_sketch.to(device = device)
        rasterized_strokes = rasterized_strokes.to(device = device)
        spatial_edge_list = spatial_edge_list.to(device = device)
        temporal_edge_list = temporal_edge_list.to(device = device)
        spatial_dist = spatial_dist.to(device = device)
        labels = labels.to(device = device)
        images = processor(rasterized_sketch.unsqueeze(0).unsqueeze(1).expand(-1, 3, -1, -1), return_tensors="pt", do_rescale=False)
        
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

        unpacked_rasterized_strokes = rasterized_strokes
        unpacked_spatial_edges = spatial_edge_list
        unpacked_temporal_edges = temporal_edge_list
        unpacked_spatial_distances = spatial_dist
        unpacked_labels = labels
        graphs = []

        stroke_pixel_sum = [(~s).reshape(num_patches, 224 // num_patches, num_patches, 224 // num_patches).sum(dim=(1, 3)) for s in unpacked_rasterized_strokes]
        features = torch.stack([(patch_features[0] * s[..., None]).sum(dim=(0, 1)) / s.sum().clamp(1) for s in stroke_pixel_sum])

        graph = HeteroData()
        graph["stroke"].x = features
        graph["stroke", "spatial", "stroke"].edge_indices = unpacked_spatial_edges.to(device = device)
        graph["stroke", "temporal", "stroke"].edge_indices = unpacked_temporal_edges.to(device = device)
        if config["graphstyle"] == "dist" or config["graphstyle"] == "dist_mini":
            dist = torch.exp(- unpacked_spatial_distances**2 / config["sigma"] ** 2)
            graph["stroke", "spatial", "stroke"].edge_weight = dist
        else:
            graph["stroke", "spatial", "stroke"].edge_weight = torch.ones(len(graph["stroke", "spatial", "stroke"].edge_indices), device = device)
        graph["stroke"].label = torch.cat([unpacked_labels, unpacked_labels[[-1],]])[:len(unpacked_rasterized_strokes)]
        graph["stroke"].logits = 0

        assert len(graph["stroke"].label) == len(graph["stroke"].x)
        
        print(graph)
        #graphs.append(graph)
        batch = graph
        
        #print(model(batch.x_dict, {k: batch.edge_indices_dict[k].T for k in batch.edge_indices_dict}))
        if config["model_type"] == "HeteroGraphConv":
            batch["stroke"].logits = model(batch.x_dict, {k: batch.edge_indices_dict[k].T for k in batch.edge_indices_dict}, edge_weight_dict = {('stroke', 'spatial', 'stroke'): batch.edge_weight_dict["stroke", "spatial", "stroke"]})["stroke"]
        else:
            batch["stroke"].logits = model(batch.x_dict, {k: batch.edge_indices_dict[k].T for k in batch.edge_indices_dict})["stroke"]
        
        # print(torch.min(batch["stroke"].label), torch.max(batch["stroke"].label), batch["stroke"].logits.shape)
        mask = batch["stroke"].label != 0
        preds = torch.argmax(batch["stroke"].logits, -1)
        accuracy = (torch.argmax(batch["stroke"].logits[mask], -1) == batch["stroke"].label[mask]).float().mean()
        preds[~mask] = 0

        indices = torch.where(vector_sketch[:, 2])[0] + 1
        strokes = torch.tensor_split(vector_sketch, indices.cpu())
        strokes = [s for s in strokes if len(s) > 0]

        #labels = batch["stroke"].label  #preds[i]#.item()
        labels = batch["stroke"].label#preds

        print(batch["stroke"].label)
        print(preds)

        unique_labels = labels.unique().tolist()
        print(unique_labels)
        colors = plt.cm.get_cmap('tab10', len(unique_labels))
        label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}
        label_to_color[0] = "black"

        all_classes = [""] + dataset.all_classes

        for l in unique_labels:
            print(all_classes[l - 1])

        handles = []
        handles.append(
                    plt.Line2D([], [], color="black", label="")  # For the legend
                )
        fig, axis = plt.subplots(1, figsize = (6, 6))
        for i, s in enumerate(strokes):
            label = labels[i].item()
            print(all_classes[label])
            if all_classes[label] not in [h.get_label() for h in handles]:  # Avoid duplicate legend entries
                handles.append(
                    plt.Line2D([], [], color=label_to_color[label], label=all_classes[label])  # For the legend
                )
            axis.plot(s[:, 0], - s[:, 1], color=label_to_color[label])#, color = cmap(cos_sim[-1, i].detach().cpu()))
        print(handles)
        axis.legend(handles=handles, title="Labels")
        axis.set_axis_off()
        plt.tight_layout()
        plt.savefig("test.pdf", dpi = 900)
        print(accuracy)
        
        

