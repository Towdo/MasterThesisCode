import numpy
import matplotlib.pyplot as plt
import PIL

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import os

from tqdm import tqdm
from fscocoloader import FSCOCOLoader
from quickdrawloader import QuickDrawLoader
from geometryset import GeometrySet

from augmentations import randomStrokePerspective, randomHorizontalFlip, addNoise

from hyper_lstm import HyperLSTM

from sketch_rnn import SketchRNNEncoder, SketchRNN

import hashlib
import os

from functools import partial
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import ray.cloudpickle as pickle
import tempfile
from lockfile import Lock

from pathlib import Path

from collections import defaultdict

import time


device = "cuda"
eps = 1e-7
maxeps = 10

class ClassifierTransformer(nn.Module):
    def __init__(self, config):
        super(ClassifierTransformer, self).__init__()

        self.config = config

        # encoder_layer = TransformerEncoderLayer(d_model=hp.d_model, nhead=hp.nhead, dim_feedforward=hp.dim_feedforward, dropout=hp.dropout, batch_first=True)
        # self.transformer = TransformerEncoder(encoder_layer, num_layers = hp.num_layers)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=config["d_model"], nhead=config["nhead"], dim_feedforward=config["dim_feedforward"], dropout=config["dropout"], batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers = config["num_layers"])
        self.linear = nn.Linear(config["d_model"], 1)

        if config["do_pos_embed"]:
            # self.pos_emb = nn.Parameter(torch.empty([500, config["d_model"]], device = device))
            self.pos_emb = nn.Parameter(torch.empty(500, device = device))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, inp, mask):
        batch_size = inp.shape[0]
        num_strokes = inp.shape[1]
        if self.config["do_pos_embed"]:
            inp = inp + self.pos_emb[:num_strokes].unsqueeze(1)#.expand(batch_size, -1).unsqueeze(0)#torch.concatenate([inp, self.pos_emb[:num_strokes].unsqueeze(0).repeat(batch_size, 1, 1)], dim = 2)

        outp = self.transformer(inp, src_key_padding_mask = ~mask)

        if self.config["pooling"] == "element":
            """Get embedding of first element only:"""
            outp = self.linear(outp[:, 0])
        elif self.config["pooling"] == "average":
            """Global average pooling"""
            outp = self.linear(torch.sum(outp * mask.unsqueeze(-1), dim = 1) / torch.sum(mask, dim = 1).unsqueeze(-1))#torch.mean(outp, dim = 1))
        elif self.config["pooling"] == "max":
            """Global max pooling"""
            zero_mask = mask == 0
            zero_mask = zero_mask.unsqueeze(-1).expand_as(outp)
            outp[zero_mask] = -float("Inf")
            outp = self.linear(torch.max(outp, dim = 1)[0])

        assert not outp.isnan().any()
        return outp.squeeze(1)

def run_epoch(config, model, encoder, optimizers, dataloader, name):
    avg_L = 0
    avg_acc = 0
    for i, d in enumerate(dataloader):
        if optimizers is not None:
            for opt in optimizers:
                opt.zero_grad()

        ids, strokes, lengths, categories = d
        strokes = strokes.to(device = device, dtype = torch.float32)
        if config["do_augment"]:
            # FIXME: We augment each batch with the same augmentation
            batch, max_strokes, max_points, _ = strokes.shape
            strokes = strokes.reshape(batch * max_strokes, max_points, 5)
            strokes = addNoise(strokes)
            strokes = randomHorizontalFlip(strokes)
            strokes = randomStrokePerspective(strokes)
            strokes = strokes.reshape(batch, max_strokes, max_points, 5)

        batch_size, num_strokes, num_points, _ = strokes.shape
        strokes = strokes.reshape(batch_size * num_strokes, num_points, 5)
        flat_lengths = lengths.clamp(min = 1).flatten()  #FIXME: Clamp is not wanted here

        strokes = torch.nn.utils.rnn.pack_padded_sequence(strokes, flat_lengths, batch_first = True, enforce_sorted = False)
        latent, _, _ = encoder(strokes)

        latent = latent.reshape(batch_size, num_strokes, config["transformer"]["d_model"])

        output = model(latent, (lengths > 0).to(device = device))

        label = torch.zeros(batch_size, dtype = torch.float32)
        for j, cat in enumerate(categories):
            label[j] = 1 if config["object"] in cat else 0
        label = label.to(device = device)

        # print(output)
        # print(label)
        loss = torch.nn.BCEWithLogitsLoss()(output, label)
        
        avg_L += loss.item()
        avg_acc += ((output > 0.5) == label).float().mean().item()

        if optimizers is not None:
            loss.backward()
            # gradient cliping
            # nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite = True)
            for opt in optimizers:
                opt.step()

    return {"L": avg_L / len(dataloader), "acc": avg_acc / len(dataloader)}


def train_transformer(config):
    # with Lock("./lock.lock"):
    #     print("Preload")
    dataset = torch.load(config["dataset_path"])
        # print("Postload")
    train_dataset, test_dataset, _ = torch.utils.data.random_split(dataset, [0.8 * config["dataset_perc"], 0.2 * config["dataset_perc"], 1 - config["dataset_perc"]])

    def collate_fn(batch):
        index, sketch, lengths, categories = zip(*batch)
        return torch.utils.data.dataloader.default_collate(index), torch.utils.data.dataloader.default_collate(sketch), torch.utils.data.dataloader.default_collate(lengths), categories

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, collate_fn = collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = config["batch_size"], shuffle = True, collate_fn = collate_fn)

    torch.autograd.set_detect_anomaly(True)

    model = ClassifierTransformer(config["transformer"]).to(device = device)

    # def get_n_params(model):
    #     pp=0
    #     for p in list(model.parameters()):
    #         nn=1
    #         for s in list(p.size()):
    #             nn = nn*s
    #         pp += nn
    #     return pp
    # print("N_PARAMS:", get_n_params(model), config)
    # return

    with open(os.path.expanduser("~/code/" + config["encoder_name"] + "/checkpoint_000000/data.pkl"), "rb") as fp:
        encoder_checkpoint = pickle.load(fp)
    encoder_config = encoder_checkpoint["config"]["model"]
    sketchrnn = SketchRNN(encoder_config)                               #FIXME: Don't need decoder here
    sketchrnn.load_state_dict(encoder_checkpoint["net_state_dict"])
    encoder = sketchrnn.encoder.to(device = device)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            if config["train_encoder"]:
                encoder.load_state_dict(checkpoint_state["encoder_state_dict"])
            encoder.load_state_dict
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    optimizers = [optim.Adam(model.parameters(), lr = config["lr"])]
    if config["train_encoder"]:
        optimizers += [optim.Adam(encoder.parameters(), lr = config["encoder_lr"])]

    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[0], factor = config["lr_decay"], patience = config["lr_patience"], min_lr = config["lr_min"])
    model.train()

    for epoch in range(start_epoch, config["epochs"]):
        start = time.time()
        model.train()
        train_loss_info = run_epoch(config, model, encoder, optimizers, train_dataloader, "TRAIN")
        scheduler.step(train_loss_info["L"])
        model.eval()
        test_loss_info = run_epoch(config, model, encoder, None, test_dataloader, "TEST")
        
        if epoch % 300 == 0 and config["create_checkpoints"] and epoch > 0:
            checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": model.state_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "optimizer_state_dict": [opt.state_dict for opt in optimizers],
                "config": config
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)

        
                train.report(
                    {
                    "train_L": train_loss_info["L"],
                    "train_acc": train_loss_info["acc"],
                    "test_L": test_loss_info["L"],
                    "test_acc": test_loss_info["acc"],
                    },
                    checkpoint = checkpoint
                )
        else:
            train.report(
                    {
                    "train_L": train_loss_info["L"],
                    "train_acc": train_loss_info["acc"],
                    "test_L": test_loss_info["L"],
                    "test_acc": test_loss_info["acc"],
                    },
                )



if __name__ == "__main__":
    import sys
    myconfig = {
        "transformer": {
            "d_model": int(sys.argv[2]),
            "nhead": 4, #tune.choice([1, 4]),
            "pooling": tune.grid_search(["element", "average", "max"]),
            "dim_feedforward": tune.grid_search([2048, 4096, 4096*2]),#tune.grid_search([32, 128, 512]),
            "dropout": 0.0, #tune.choice([0.0, 0.1, 0.2, 0.5]),
            "num_layers": 8,#tune.grid_search([1, 2]),
            "do_pos_embed": True, #tune.choice([0, 4]),
        },
        "batch_size": 100,
        "train_encoder": bool(int(sys.argv[3])),#tune.grid_search([True, False]),
        "encoder_lr": 0.001,
        "lr": 0.01,
        "lr_decay": 0.1,
        "lr_patience": 50,    #FIXME: DEBUG
        "lr_min": 0.0001,
        "epochs": 301,
        "dataset_perc": 1.0,
        "create_checkpoints": True,
        "num_examples": None,
        "dataset": "fscoco",
        "dataset_path": "/home/s6toweis_hpc/code/classification_data.pickle",
        "object": "tree",
        "encoder_name": sys.argv[1],#"checkpoint_0.1_4",
        "do_augment": True,
    }

    print("Config")
    print(myconfig)

    with tempfile.TemporaryDirectory() as dataset_dir:
        if myconfig["dataset_path"] is None:
            dataset_path = Path(dataset_dir) / "data.pkl"
            print("Started generating dataset")
            if myconfig["dataset"] == "fscoco":
                dataset = FSCOCOLoader(mode = "normal", max_length = None, get_annotations = True, do_normalise = True, do_simplify = True, num_examples = myconfig["num_examples"])
            elif myconfig["dataset"] == "geometry":
                # dataset = GeometrySet(mode = "normal", size = 5000, num_distribution = [0.1, 0.2, 0.3, 0.4], shape_distribution = [0.0, 0.3, 0.3, 0.4], noise = 5)
                dataset = GeometrySet(mode = "normal", size = 10000, num_distribution = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], shape_distribution = [0.3, 0.1, 0.3, 0.3], noise = 5.)
                # occurences = dataset.get_label_occurences(myconfig["object"])
                # print("Occurences", occurences, "/", len(dataset), "(", occurences * 1.0 / len(dataset), ")")
            print("Presave")
            torch.save(dataset, dataset_path)
            print("Postsave")

            myconfig["dataset_path"] = dataset_path

        scheduler = ASHAScheduler(
            metric="test_acc",
            mode="max",
            max_t=501,
            grace_period=101,
            reduction_factor=3,
        )

        # class EarlyStopper(tune.stopper.Stopper):
        #     def __init__(self, metric, num_wait):
        #         self._metric = metric
        #         self._min = defaultdict(lambda: None)
        #         self._num_wait = num_wait
        #         self._counter = defaultdict(lambda: 0)

        #     def __call__(self, trial_id, result):
        #         if self._min[trial_id] is None:
        #             self._min[trial_id] = result[self._metric]
        #             self._counter[trial_id] = 0
        #             return False
                
        #         if self._min[trial_id] > result[self._metric]:
        #             self._min[trial_id] = result[self._metric]
        #             self._counter[trial_id] = 0
        #         else:
        #             self._counter[trial_id] += 1

        #         if self._counter[trial_id] > self._num_wait:
        #             return True
                
        #         return False

        #     def stop_all(self):
        #         return False

        # train_transformer(myconfig)
        result = tune.run(
            train_transformer,
            resources_per_trial={"cpu": 32, "gpu": 1.0},
            config=myconfig,
            num_samples = 1,
            verbose = 1,
            # scheduler = scheduler,
            #time_budget_s=36000, #10 hours
            # stop=EarlyStopper("test_L", 8)
        )

