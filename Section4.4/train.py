import torch
import numpy as np

from data import dinoSketchSet, dinoSketchSetAnnotated
# from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt

import torch.nn.functional as F
import torchvision.transforms as T

from torch_geometric.nn import SimpleConv
from torch_geometric.data import Batch

import torch.nn.utils.rnn as rnn

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

import tempfile

from tqdm import tqdm

from test_clip_supervised import train_f

# from douglaspeucker import DouglasPeucker
import os

from transformers import CLIPImageProcessor, CLIPVisionModel

upscale = 1
device = "cuda"

from collections import defaultdict

class SketchDeepWalk:
    def __init__(self, walk_length=None, num_walks=None):
        self.walk_length = walk_length
        self.num_walks = num_walks

    def simulate_walks(self, edge_index, num_nodes):
        # Step 1: Convert edge_index to adjacency list using numpy for efficiency
        adj = defaultdict(list)
        row, col = edge_index
        for u, v in zip(row, col):
            adj[u.item()].append(v.item())
            adj[v.item()].append(u.item())
        
        # Step 2: Generate walks
        walks = np.zeros((self.num_walks * num_nodes, self.walk_length), dtype=np.int64)
        
        for walk_id in range(self.num_walks * num_nodes):
            start_node = walk_id % num_nodes #np.random.randint(num_nodes)
            current_walk = [start_node]
            
            for i in range(1, self.walk_length):
                current_node = current_walk[-1]
                if current_node not in adj or len(adj[current_node]) == 0:
                    break  # End walk early if no neighbors
                next_node = np.random.choice(adj[current_node])  # Random neighbor
                current_walk.append(next_node)
            
            # Store walk
            walks[walk_id, :len(current_walk)] = current_walk
        
        return torch.tensor(walks)



def train_epoch(loader, optim, image_transforms, get_features, get_patches, num_patches, feature_dim, model, use_temporal, temperature=0.07, walk_length = None, num_walks = None, context_size = None, num_neg = None):
    deepwalk = SketchDeepWalk(walk_length=walk_length, num_walks=num_walks)
    mean_loss = 0
    
    for data in tqdm(loader, "Training", disable=True):
        rasterized_sketch, rasterized_strokes, spatial_edge_list, temporal_edge_list = data
        rasterized_sketch = rasterized_sketch.to(device=device)
        rasterized_strokes = rasterized_strokes.to(device=device)
        spatial_edge_list = spatial_edge_list.to(device=device)
        temporal_edge_list = temporal_edge_list.to(device=device)

        batch_size = len(rasterized_sketch)
        
        image = image_transforms(rasterized_sketch.float().unsqueeze(1).expand(batch_size, 3, 224, 224))
        features = get_features(image)
        feature_grid = features.reshape(batch_size, num_patches, num_patches, feature_dim)

        unpacked_strokes = torch.nn.utils.rnn.unpack_sequence(rasterized_strokes)
        unpacked_spatial_edges = torch.nn.utils.rnn.unpack_sequence(spatial_edge_list)
        unpacked_temporal_edges = torch.nn.utils.rnn.unpack_sequence(temporal_edge_list)

        loss = torch.tensor([0.], device=device)

        for i in range(batch_size):
            example_loss = torch.tensor([0.], device=device)
            num_strokes = len(unpacked_strokes[i])
            
            # Calculate stroke features
            stroke_pixel_sum = [(~s).reshape(14, 16, 14, 16).sum(dim=(1, 3)) for s in unpacked_strokes[i]]
            stroke_features = torch.stack([(feature_grid[i] * s[..., None]).sum(dim=(0, 1)) / s.sum().clamp(1) for s in stroke_pixel_sum])
            #stroke_features = model(stroke_features)
            stroke_features = F.normalize(stroke_features)

            # Get edges and generate walks
            if use_temporal:
                edges = torch.cat([unpacked_spatial_edges[i], unpacked_temporal_edges[i]]).T
            else:
                edges = unpacked_spatial_edges[i].T
                
            if edges.shape[1] == 0:
                continue

            # Generate random walks using DeepWalk
            walks = deepwalk.simulate_walks(edges, num_strokes)
            
            # For each walk, compute loss using skip-gram with negative sampling
            for walk in walks:
                for pos in range(len(walk)):
                    # Context window around current position
                    # context_size = context_size
                    start = max(0, pos - context_size)
                    end = min(len(walk), pos + context_size + 1)
                    
                    # Context nodes
                    context_indices = list(range(start, pos)) + list(range(pos + 1, end))
                    if not context_indices:
                        continue
                        
                    target = walk[pos]
                    target_features = stroke_features[target]
                    
                    # Positive samples
                    pos_features = stroke_features[walk[context_indices]]
                    pos_sim = torch.matmul(target_features, pos_features.T) / temperature
                    pos_loss = -torch.log(torch.sigmoid(pos_sim)).mean()
                    
                    # Negative sampling
                    # num_neg = 2
                    neg_indices = torch.randint(0, num_strokes, (len(context_indices), num_neg), device=device)
                    neg_features = stroke_features[neg_indices]
                    neg_sim = torch.matmul(target_features.unsqueeze(0), neg_features.transpose(1, 2)).squeeze()
                    neg_loss = -torch.log(1 - torch.sigmoid(neg_sim)).mean()
                    
                    example_loss += (pos_loss + neg_loss)
            num_pairs = sum(len(walk) * (2 * context_size) for walk in walks)
            loss += example_loss / num_pairs
        optim.zero_grad()
        loss.backward()
        mean_loss += loss.item()
        optim.step()

    return mean_loss / len(loader)

def test_epoch(test_data, loader, image_transforms, get_features, get_patches, num_patches, feature_dim, model, use_temporal, temperature=0.07, walk_length = None, num_walks = None, context_size = None, num_neg = None):
    deepwalk = SketchDeepWalk(walk_length=walk_length, num_walks=num_walks)
    with torch.no_grad():
        mean_loss = 0
        total_batches = 0
        feature_dict = {}
        
        for data in tqdm(loader, "Testing", disable=False):
            rasterized_sketch, rasterized_strokes, spatial_edge_list, temporal_edge_list, labels = data
            rasterized_sketch = rasterized_sketch.to(device=device)
            rasterized_strokes = rasterized_strokes.to(device=device)
            
            batch_size = len(rasterized_sketch)
            
            image = image_transforms(rasterized_sketch.float().unsqueeze(1).expand(batch_size, 3, 224, 224))
            features = get_features(image)
            feature_grid = features.reshape(batch_size, num_patches, num_patches, feature_dim)

            unpacked_strokes = torch.nn.utils.rnn.unpack_sequence(rasterized_strokes)
            unpacked_spatial_edges = torch.nn.utils.rnn.unpack_sequence(spatial_edge_list)
            unpacked_temporal_edges = torch.nn.utils.rnn.unpack_sequence(temporal_edge_list)
            unpacked_labels = torch.nn.utils.rnn.unpack_sequence(labels)

            batch_loss = 0

            for i in range(batch_size):
                example_loss = 0
                num_strokes = len(unpacked_strokes[i])
                
                stroke_pixel_sum = [(~s).reshape(14, 16, 14, 16).sum(dim=(1, 3)) for s in unpacked_strokes[i]]
                stroke_features = torch.stack([(feature_grid[i] * s[..., None]).sum(dim=(0, 1)) / s.sum().clamp(1) for s in stroke_pixel_sum])
                #stroke_features = model(stroke_features)
                stroke_features = F.normalize(stroke_features)
                
                stroke_labels = unpacked_labels[i]
                for j, label in enumerate(stroke_labels):
                    label_key = label.item()
                    if label_key not in feature_dict:
                        feature_dict[label_key] = []
                    feature_dict[label_key].append(stroke_features[j].cpu())

                if use_temporal:
                    edges = torch.cat([unpacked_spatial_edges[i], unpacked_temporal_edges[i]]).T
                else:
                    edges = unpacked_spatial_edges[i].T
                
                if edges.shape[1] == 0:
                    continue

                walks = deepwalk.simulate_walks(edges.cpu(), num_strokes)
                
                for walk in walks:
                    for pos in range(len(walk)):
                        # context_size = 3
                        start = max(0, pos - context_size)
                        end = min(len(walk), pos + context_size + 1)
                        
                        context_indices = list(range(start, pos)) + list(range(pos + 1, end))
                        if not context_indices:
                            continue
                            
                        target = walk[pos]
                        target_features = stroke_features[target]
                        
                        pos_features = stroke_features[walk[context_indices]]
                        pos_sim = torch.matmul(target_features, pos_features.T) / temperature
                        pos_loss = -torch.log(torch.sigmoid(pos_sim)).mean()
                        
                        # num_neg = 3
                        neg_indices = torch.randint(0, num_strokes, (len(context_indices), num_neg), device=device)
                        neg_features = stroke_features[neg_indices]
                        neg_sim = torch.matmul(target_features.unsqueeze(0), neg_features.transpose(1, 2)).squeeze()
                        neg_loss = -torch.log(1 - torch.sigmoid(neg_sim)).mean()
                        
                        example_loss += (pos_loss + neg_loss)
            num_pairs = sum(len(walk) * (2 * context_size) for walk in walks)
            batch_loss += example_loss / num_pairs

            mean_loss += batch_loss.item()
            total_batches += 1

        training_style_loss = mean_loss / total_batches if total_batches > 0 else 0
        
        feature_dict = {k: torch.stack(v) for k, v in feature_dict.items() if len(v) > 0}
        feature_dict.pop(0, None)

        class_centers = {k: torch.mean(v, dim=0) for k, v in feature_dict.items()}
        
        all_logits = []
        all_labels = []

        for label, features in feature_dict.items():
            for feature in features:
                logits = torch.stack([torch.matmul(feature, center) for center in class_centers.values()])
                all_logits.append(logits)
                all_labels.append(label)

        all_logits = torch.stack(all_logits).to(device=device)
        all_labels = torch.tensor(all_labels, device=device)
        all_labels = all_labels - 1

        class_loss = F.cross_entropy(all_logits / temperature, all_labels)
        
        num_classes = len(feature_dict)
        mean_inter_similarity = torch.zeros(num_classes, num_classes, device=device)
        
        for i, (key1, features1) in enumerate(feature_dict.items()):
            features1 = F.normalize(features1, dim=1)
            for j, (key2, features2) in enumerate(feature_dict.items()):
                features2 = F.normalize(features2, dim=1)
                mean_inter_similarity[i, j] = (features1 @ features2.T).mean()

        old_class_loss = F.cross_entropy(mean_inter_similarity / temperature, torch.arange(len(mean_inter_similarity), device=device))

        return training_style_loss, class_loss.item(), old_class_loss.item(), mean_inter_similarity.cpu()


def walk_trainer(config):
    print(config)
    # simplify = lambda  x: DouglasPeucker(x, 2)
    data = dinoSketchSet(os.path.expanduser("~/data/fscoco-clone/"), graphstyle=config["graphstyle"])#, pre_transform = simplify, segmentation = False)
    #test_data = SketchGraph(os.path.expanduser("~/data/fscoco/"), pre_transform = simplify, segmentation = True)
    # test_data = dinoSketchSetAnnotated(os.path.expanduser("~/data/fscoco/"), pre_transform = simplify, segmentation = True, start_index = 500, num_examples = 475)
    test_data = dinoSketchSetAnnotated(os.path.expanduser("~/data/fscoco-clone/"), graphstyle=config["graphstyle"], min_occurrences = 20)

    def collate_fn_train(data):
        rasterized_sketch, rasterized_strokes, spatial_edge_list, temporal_edge_list = zip(*data)
        return torch.utils.data.default_collate(rasterized_sketch), rnn.pack_sequence(rasterized_strokes, enforce_sorted=False), rnn.pack_sequence(spatial_edge_list, enforce_sorted=False), rnn.pack_sequence(temporal_edge_list, enforce_sorted=False)
    
    def collate_fn_test(data):
        rasterized_sketch, rasterized_strokes, spatial_edge_list, temporal_edge_list, labels = zip(*data)
        return torch.utils.data.default_collate(rasterized_sketch), rnn.pack_sequence(rasterized_strokes, enforce_sorted=False), rnn.pack_sequence(spatial_edge_list, enforce_sorted=False), rnn.pack_sequence(temporal_edge_list, enforce_sorted=False), rnn.pack_sequence(labels, enforce_sorted=False)

    loader = torch.utils.data.DataLoader(data, batch_size = 1, shuffle = True, collate_fn = collate_fn_train)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = True, collate_fn = collate_fn_test) #FIXME SHUFFLE

    # def generate_weights(graph):
    #     weights = torch.zeros(graph.edge_types.shape, device = device)
    #     sigma = 1
    #     weights[graph.edge_types == 0] = torch.exp(- graph.edge_distances[graph.edge_types == 0]**2 / sigma**2)
    #     weights[graph.edge_types == 1] = 0.1
    #     assert (weights >= 0).all()
    #     return weights

    if config["vision_model"] == "dino":
        vision_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device = device)

        image_transforms = T.Compose([
            T.Resize((224 * upscale, 224 * upscale), interpolation=T.InterpolationMode.BILINEAR),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        get_features = lambda image: vision_model.forward_features(image)["x_norm_patchtokens"]

        get_patches = lambda graph: graph.patches_16

        num_patches = 16
    elif config["vision_model"] == "clip":
        vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch16",
            # "openai/clip-vit-large-patch14",
            device_map=device,
        )

        feature_dim = 768

        clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

        image_transforms = lambda x : clip_processor(images = x, return_tensors="pt")

        get_features = lambda inputs: vision_model(pixel_values = inputs["pixel_values"].to(device = device), output_hidden_states = True)["hidden_states"][config["clip_layer"]][:, 1:]
        
        get_patches = lambda graph: graph.patches_14

        num_patches = 14

    # # Freeze all parameters
    # for param in vision_model.parameters():
    #     param.requires_grad = False

    # # Unfreeze just the projection
    # vision_model.vision_model.encoder.layers[config["clip_layer"] - 1].mlp.fc2.weight.requires_grad = True

    # optim = torch.optim.Adam([vision_model.vision_model.encoder.layers[config["clip_layer"] - 1].mlp.fc2.weight], lr = config["lr"])

    optim = torch.optim.Adam(vision_model.parameters(), lr = config["lr"])

    train_losses = []
    test_losses = []

    test_loss, class_loss, old_class_loss, _ = test_epoch(test_data, test_loader, image_transforms, get_features, get_patches, num_patches, feature_dim, None, config["use_temporal"], config["temperature"], config["walk_length"], config["num_walks"], config["context_size"], config["num_neg"])
    print("Start train_f")
    _, _, test_accuracy = train_f(vision_model, config["classif_config"])
    print("End train_f")
    print("Baseline:", test_loss, class_loss, old_class_loss, test_accuracy)

    train.report(
                    {
                    "train_L": 0.,
                    "test_L": test_loss,
                    "class_acc": test_accuracy,
                    },
                )

    EPOCHS = config["epochs"]
    for epoch in tqdm(range(EPOCHS)):
        
        train_loss = train_epoch(loader, optim, image_transforms, get_features, 
                        get_patches, num_patches, feature_dim, None, config["use_temporal"],
                        config["temperature"], config["walk_length"], config["num_walks"], config["context_size"], config["num_neg"])
        train_losses.append(train_loss)

        test_loss, class_loss, old_class_loss, mean_inter_sim = test_epoch(test_data, test_loader, image_transforms, get_features, get_patches, num_patches, feature_dim, None, config["use_temporal"], config["temperature"], config["walk_length"], config["num_walks"], config["context_size"], config["num_neg"])
        test_losses.append(test_loss)

        print("Start train_f")
        _, _, test_accuracy = train_f(vision_model, config["classif_config"])
        print("End train_f")

        print("Test accuracy", test_accuracy)
        train.report(
                    {
                    "train_L": train_loss,
                    "test_L": test_loss,
                    "class_acc": test_accuracy,
                    },
                )
        

        # N = len(mean_inter_sim)
        # # print(train_loss, test_loss, class_loss, old_class_loss, torch.diag(mean_inter_sim).mean().item(), torch.triu(mean_inter_sim, diagonal = 1).sum().item() / (N * (N - 1) / 2))
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            # Save model state
            checkpoint_path = os.path.join(temp_checkpoint_dir, "model.pt")
            torch.save({
                'model_state_dict': vision_model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'config': config,
                'feature_dim': feature_dim,
            }, checkpoint_path)
            
            # Create final checkpoint for Ray
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {
                    "train_L": train_loss,
                    "test_L": test_loss,
                    "class_acc": test_accuracy,
                },
                checkpoint=checkpoint
            )
            
    # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
    #     # Save model state
    #     checkpoint_path = os.path.join(temp_checkpoint_dir, "model.pt")
    #     torch.save({
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optim.state_dict(),
    #         'config': config,
    #         'feature_dim': feature_dim,
    #     }, checkpoint_path)
        
    #     # Create final checkpoint for Ray
    #     checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
    #     train.report(
    #         {
    #             "train_L": train_loss,
    #             "test_L": test_loss,
    #             "class_loss": class_loss,
    #             "old_class_loss": old_class_loss,
    #             "self_sim": torch.diag(mean_inter_sim).mean().item(),
    #             "trans_sim": torch.triu(mean_inter_sim, diagonal = 1).sum().item() / (N * (N - 1) / 2),
    #         },
    #         checkpoint=checkpoint
    #     )

if __name__ == "__main__":
    import sys

    classif_config = {
        "batch_size": 5,#tune.grid_search([1, 5, 10, 20]),#tune.grid_search([20, 40, 60, 80, 20, 30, 40, 50]),
        "graphstyle": sys.argv[1],#tune.grid_search(["sym_bbox", "asym_bbox", "dist"]),
        "vit": "clip",#tune.grid_search(["clip", "dino"]),#"clip","dino"
        "model_type": "HeteroGraphConv",#"Linear", "HeteroGraphConv", "HeteroGAT"
        "clip_level": 10,#tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        "lr": 0.0001,
        "weight_decay": 0.001,
        "sigma": 4,
        "num_layers": 2,
        "hardmode": True,
        "epochs": 32
    }

    myconfig = {
        "lr": 0.000001,#tune.choice([0.001, 0.0001, 0.00001]),
        "num_samples": 20,
        "epochs": 1,
        "vision_model": "clip",
        "temperature": 1.,#tune.grid_search([0.1, 0.5, 1.]),
        "clip_layer": 10,
        "use_temporal": True,#tune.grid_search([True, False]),
        "graphstyle": sys.argv[1],# tune.grid_search(["sym_bbox", "asym_bbox", "dist"])
        "walk_length": 8,
        "num_walks": 1,
        "context_size": int(sys.argv[2]),
        "num_neg": tune.grid_search([1, 4, 16, 32]),#tune.grid_search([5, 10]),
        "classif_config": classif_config,
        }

    tune.run(
        walk_trainer,
        resources_per_trial={"cpu": 16, "gpu": 1},
        config=myconfig,
        num_samples=1,
    )

    # trainer(myconfig)
    
    # scheduler = ASHAScheduler(
    #         metric="test_L",
    #         mode="min",
    #         max_t=myconfig["epochs"],
    #         grace_period=25,
    #         reduction_factor=2,
    #     )

    # result = tune.run(
    #     trainer,
    #     resources_per_trial={"cpu": 16, "gpu": 1.0},
    #     config=myconfig,
    #     num_samples=myconfig["num_samples"],
    #     verbose = 1,
    #     # stop=EarlyStopper("test_LR", 20)
    #     scheduler=scheduler,
    #     #stop=tune.stopper.TrialPlateauStopper("test/L", std = 0.001)
    # )