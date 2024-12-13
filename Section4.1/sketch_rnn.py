import numpy
import matplotlib.pyplot as plt
import PIL

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
import os

from tqdm import tqdm
from fscocoloader import FSCOCOLoader
from quickdrawloader import QuickDrawLoader
from geometryset import GeometrySet

from augmentations import randomStrokePerspective, randomHorizontalFlip, addNoise

from utils import recursive_add_arguments, recursive_update_config

from hyper_lstm import HyperLSTM

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

from pathlib import Path

from collections import defaultdict

import time

import argparse


device = "cuda"
eps = 1e-7
maxeps = 10


#################################

class SketchRNNEncoder(nn.Module):
    def __init__(self, config):
        super(SketchRNNEncoder, self).__init__()

        self.config = config
        # bidirectional lstm:
        self.lstm = nn.LSTM(5, config["enc_hidden_size"], bidirectional=True, batch_first = True)
        self.dropout = nn.Dropout(config["dropout"])
        # create mu and sigma from lstm's last output:
        self.fc_mu = nn.Linear(2*config["enc_hidden_size"], config["Nz"])
        self.fc_sigma = nn.Linear(2*config["enc_hidden_size"], config["Nz"])
        # active dropout:
        self.train()

    def forward(self, inputs, hidden_cell = None):
        config = self.config

        if hidden_cell is None:
            batch_sizes = inputs.batch_sizes
            max_batch_size = batch_sizes[0].item()  # Get the maximum batch size

            hidden = torch.zeros(2, max_batch_size, config["enc_hidden_size"], device = device)
            cell = torch.zeros(2, max_batch_size, config["enc_hidden_size"], device = device)
            hidden_cell = (hidden, cell)
        _, (hidden, cell) = self.lstm(inputs, hidden_cell)
        hidden = self.dropout(hidden)   #FIXME: ?
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = hidden
        hidden_cat = torch.cat([hidden_forward, hidden_backward], dim = 1)
        # mu and sigma:
        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp((sigma_hat/2.).clamp(max = maxeps))#.clamp(max = maxeps)
        # N ~ N(0,1)
        N = torch.normal(torch.zeros(mu.shape, device = device), torch.ones(mu.shape, device = device))
        z = mu + sigma * N
        return z, mu, sigma_hat
    

#############
class SketchRNNDecoder(nn.Module):
    def __init__(self, config):
        super(SketchRNNDecoder, self).__init__()

        self.config = config
        # to init hidden and cell from z:
        self.fc_hc = nn.Linear(config["Nz"], 2*config["dec_hidden_size"])
        # unidirectional lstm:
        self.lstm = nn.LSTM(config["Nz"]+5, config["dec_hidden_size"], batch_first = True)

        self.dropout = nn.Dropout(config["dropout"])
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(config["dec_hidden_size"],6*config["M"]+3)


    def forward(self, inputs, z, hidden_cell = None):
        config = self.config
        
        if hidden_cell is None:
            hidden, cell = torch.split(F.tanh(self.fc_hc(z)), config["dec_hidden_size"], 1) #FIXME
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        outputs, lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first = True)
        outputs = self.dropout(outputs) #FIXME: ?

        #FIXME
        y = self.fc_params(outputs)
        # if self.training:
        #     y = self.fc_params(outputs)     #FIXME!!!!
        # else:
        #     y = self.fc_params(hidden)       #FIXME: [0] only for hyperlstm

        y, params_pen = y[:, :, :-3], y[:, :, -3:]
        pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy = torch.split(y, config["M"], dim = 2)
        
        pi = F.softmax(pi, dim = 2)
        mu = torch.stack([mu_x, mu_y])
        sigma = torch.exp(torch.stack([sigma_x, sigma_y]).clamp(max = maxeps))  #FIXME: DEBUG sigma does sometimes explode for no apparent reason.
        rho_xy = torch.tanh(rho_xy)
        q = F.softmax(params_pen, dim = 2)

        return pi,mu,sigma,rho_xy,q,hidden,cell

class SketchRNN(nn.Module):
    def __init__(self, config):
        super(SketchRNN, self).__init__()
        
        self.config = config
        # print("config")

        self.encoder = SketchRNNEncoder(config)
        self.decoder = SketchRNNDecoder(config)
    
    def forward(self, input):
        config = self.config

        z, mu, sigma = self.encoder(input)

        LKL = kullback_leibler_loss(mu, sigma)

        batch_size = input.batch_sizes[0].item()

        sos = torch.stack([torch.Tensor([0,0,1,0,0])]*batch_size).unsqueeze(1).to(device = device)

        # had sos at the begining of the batch:
        input, lengths = torch.nn.utils.rnn.pad_packed_sequence(input, batch_first = True)
        batch_init = torch.cat([sos, input], dim = 1)
        # expend z to be ready to concatenate with inputs:
        Nmax = input.shape[1]

        if self.config["deterministic"]:
            # print("MU")
            z = mu
        z_stack = torch.stack([z]*(Nmax+1), dim = 1)
        # inputs is concatenation of z and batch_inputs
        inputs = torch.cat([batch_init, z_stack],2)

        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths + 1, batch_first = True, enforce_sorted = False)

        # decode:
        pi, mu, sigma, rho_xy, q, _, _ = self.decoder(inputs, z)

        max_latent_dist = torch.max(torch.mean((z.unsqueeze(0) - z.unsqueeze(1))**2, dim = 2)).item()
        num_gaussian_mix = len(torch.unique(torch.argmax(pi, dim = -1).to(dtype = torch.long)))

        # make target
        batch_size = len(input)
        eos = torch.stack([torch.Tensor([0,0,0,0,1])] * batch_size).unsqueeze(1).to(device = device)
        target = torch.cat([input, eos], 1)
        dx = torch.stack([target[:, :, 0]]*config["M"], -1)
        dy = torch.stack([target[:, :, 1]]*config["M"], -1)
        mask = torch.zeros(batch_size, Nmax+1, device = device)
        for i in range(batch_size):
            mask[i, :lengths[i]] = 1
        p = target[:, :, 2:]

        LS, LP = reconstruction_loss(mask, dx, dy, p, pi, mu, sigma, rho_xy, q)

        CE = torch.sum(pi * mask.unsqueeze(-1) * ((dx - mu[0])**2 + (dy - mu[1])**2))/torch.sum(mask)   #COORDINATE ERROR TO DEBUG

        return (LS, LP, LKL), mu, sigma, pi, sigma, rho_xy, q, (CE, max_latent_dist, num_gaussian_mix)

        
    
    def conditional_generation(self, _input):
        with torch.no_grad():
            self.eval()

            z, mu, _ = self.encoder(_input)
            # print(self.config)
            if self.config["deterministic"]:
                z = mu
            sos = torch.Tensor([0,0,1,0,0]).view(1, 5).to(device = device)

            s = sos
            seq_x = []
            seq_y = []
            seq_z = []
            hidden_cell = None
            # Nmax = _input.shape[1]
            #print(s.shape, z.shape)
            for i in range(100):
                input = torch.cat([s.expand(z.shape[0], -1), z], 1).unsqueeze(0)
                input = torch.nn.utils.rnn.pack_padded_sequence(input, torch.Tensor([1]), batch_first = True, enforce_sorted = False)
                pi, mu, sigma, rho_xy, q, hidden, cell = self.decoder(input, z, hidden_cell)
                hidden_cell = (hidden, cell)

                s, dx, dy, pen_down, eos = sample_next_state(pi, mu, sigma, rho_xy, q)

                s = s.to(device = device)

                seq_x.append(dx)
                seq_y.append(dy)
                seq_z.append(pen_down)
                if eos:
                    # print("Wanted to break at", i)
                    break
            z_sample = np.array(seq_z)
            sequence = np.stack([seq_x,seq_y,z_sample]).T

            return sequence

def sample_bivariate_normal(mu,sigma,rho_xy, greedy=True, temperature = 1):
    # inputs must be floats
    if greedy:
        return mu
    sigma *= np.sqrt(temperature)
    cov = torch.Tensor([[sigma[0] * sigma[0], rho_xy * sigma[0] * sigma[1]],\
        [rho_xy * sigma[0] * sigma[1], sigma[1] * sigma[1]]])
    x = np.random.multivariate_normal(mu, cov, 1)
    return x[0][0], x[0][1]

def sample_next_state(pi, mu, sigma, rho_xy, q, temperature = 1):
    def adjust_temp(pi_pdf):
        pi_pdf = np.log(pi_pdf)/temperature
        pi_pdf -= pi_pdf.max()
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= pi_pdf.sum()
        return pi_pdf

    pi = adjust_temp(pi.flatten().cpu().numpy())
    M = pi.shape[0]
    pi_idx = np.random.choice(M, p = pi)

    q = adjust_temp(q.flatten().cpu().numpy())
    q_idx = np.random.choice(3, p = q)

    mu = mu[:, :, :, pi_idx].cpu().flatten()
    sigma = sigma[:, :, :, pi_idx].cpu()
    rho_xy = rho_xy[:, :, pi_idx].item()

    x, y = sample_bivariate_normal(mu, sigma, rho_xy)
    next_state = torch.zeros(5)
    next_state[0] = x
    next_state[1] = y
    next_state[q_idx+2] = 1

    return next_state.unsqueeze(0), x, y, q_idx == 1, q_idx == 2

def kullback_leibler_loss(mu, sigma):
    LKL = -0.5 * torch.mean(1 + sigma - mu**2 - torch.exp(sigma))
    return LKL

def bivariate_normal_pdf(dx, dy, mu, sigma, rho_xy):
    z_x = ((dx-mu[0])/(sigma[0]).clamp(min = eps))**2
    z_y = ((dy-mu[1])/(sigma[1]).clamp(min = eps))**2
    z_xy = (dx-mu[0])*(dy-mu[1])/(sigma[0]*sigma[1]).clamp(min = eps)
    z = (z_x + z_y -2*rho_xy*z_xy)
    exp = torch.exp((-z/(2*(1-rho_xy**2)).clamp(min = eps)).clamp(max = maxeps))
    norm = 2*np.pi*sigma[0]*sigma[1]*torch.sqrt((1-rho_xy**2).clamp(min = eps)).clamp(min = eps)

    return exp/norm

def bivariate_normal_logpdf(dx, dy, mu, sigma, rho_xy):
    # Is hopefully a bit more stable and easier to clamp?
    mu_x, mu_y = mu
    sigma_x, sigma_y = sigma#.clamp(max = 1000) # FIXME: This might cause trouble if not clamped?
    
    # Log of the coefficient part
    log_coeff = -torch.log((2 * torch.pi * sigma_x * sigma_y * torch.sqrt((1 - rho_xy**2).clamp(min = eps))).clamp(min = eps))

    # Exponent term
    term1 = ((dx - mu_x) / sigma_x.clamp(min = eps))**2
    term2 = ((dy - mu_y) / sigma_y.clamp(min = eps))**2
    term3 = (2 * rho_xy * (dx - mu_x) * (dy - mu_y)) / (sigma_x * sigma_y).clamp(min = eps)
    
    exponent = -1 / (2 * (1 - rho_xy**2)).clamp(min = eps) * (term1 + term2 - term3)

    return log_coeff + exponent

def reconstruction_loss(mask, dx, dy, p, pi, mu, sigma, rho_xy, q):
    # pdf = bivariate_normal_pdf(dx, dy, mu, sigma, rho_xy)
    # LS = -torch.mean(mask*torch.log(torch.sum(pi * pdf, 2).clamp(min = eps)))
    logpdf = bivariate_normal_logpdf(dx, dy, mu, sigma, rho_xy)
    
    LS = -torch.mean(mask*torch.log(torch.sum(pi * torch.exp(logpdf.clamp(max = maxeps)), 2).clamp(min = eps)).clamp(min = -maxeps, max = maxeps))
    LP = -torch.mean(p*torch.log(q.clamp(min = eps)))
    return LS, LP

def run_epoch(config, model, optimizer, dataloader, name):
    sums = {
        'LKL': 0,
        'LS': 0,
        'LP': 0,
        'L': 0,
        'CE': 0,
        'latent_dist': 0,
        'num_gaussian': 0
    }

    config["eta_step"] = 1 - (1 - config["eta_step"]) * config["R"]
    for i, inp in enumerate(dataloader):
        if optimizer is not None:
            optimizer.zero_grad()
        
        strokes, lengths = inp
        batch_size = strokes.shape[0]
        inp = strokes.to(device = device, dtype = torch.float32)
        if config["do_augment"]:
            inp = addNoise(inp)
            inp = randomHorizontalFlip(inp)
            inp = randomStrokePerspective(inp)
        inp = torch.nn.utils.rnn.pack_padded_sequence(inp, lengths, batch_first = True, enforce_sorted = False)
            
        loss_data, _, _, _, _, _, _, debug_info = model(inp)

        LS, LP, LKL = loss_data

        loss = LS + LP + config["wKL"] * torch.max(LKL,torch.Tensor([config["KL_min"]]).to(device = device)) * config["eta_step"]

        CE, max_latent_dist, num_gaussian_mix = debug_info

        # Accumulate sums
        sums['LKL'] += LKL.item() * batch_size
        sums['LS'] += LS.item() * batch_size
        sums['LP'] += LP.item() * batch_size
        sums['L'] += loss.item() * batch_size
        sums['CE'] += CE.item() * batch_size
        sums['latent_dist'] += max_latent_dist * batch_size
        sums['num_gaussian'] += num_gaussian_mix * batch_size
        if optimizer is not None:
            loss.backward()
            # gradient cliping
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite = True)
            
            optimizer.step()
        
    num_examples = len(dataloader.dataset)
    return {"L": sums['L'] / num_examples,
            "LR": (sums['LS'] + sums['LP']) / num_examples,
            "LS": sums['LS'] / num_examples,
            "LP": sums['LP'] / num_examples,
            "LKL": sums['LKL'] / num_examples},  \
            {"CE": sums['CE'] / num_examples,
            "latent_dist": sums['latent_dist'] / num_examples,
            "num_gaussian": sums['num_gaussian'] / num_examples}


def train_sketch_rnn(config):
    config["model"]["dec_hidden_size"] = 2 * config["model"]["enc_hidden_size"]
    import os
    if config["dataset"] == "quickdraw":
        raise Exception()
        class opt():
            root_dir = os.path.expanduser("~/data/quickdraw/")
        train_dataset = QuickDrawLoader(opt(), "cat", "train", max_seq_length = config["max_seq_length"])
        Nmax = train_dataset.Nmax
        test_dataset = QuickDrawLoader(opt(), "cat", "test", max_seq_length = Nmax)
    elif config["dataset"] == "fscoco" or config["dataset"] == "geometric":
        dataset = torch.load(config["dataset_path"])

        Nmax = dataset.Nmax
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    else:
        raise Exception()

    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = config["batch_size"], shuffle = True)

    model = SketchRNN(config["model"]).to(device = device)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    optimizer = optim.Adam(model.parameters(), lr = config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = config["lr_decay"], patience = config["lr_patience"], min_lr = config["lr_min"])
    model.train()

    config["eta_step"] = config["eta_min"]
    for epoch in range(start_epoch, config["epochs"]):
        start = time.time()
        model.train()
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        #     with record_function("model_inference"):
        train_loss_info, train_debug_info = run_epoch(config, model, optimizer, train_dataloader, "TRAIN")
        # print(prof.key_averages())
        scheduler.step(train_loss_info["L"])
        model.eval()
        test_loss_info, test_debug_info = run_epoch(config, model, None, test_dataloader, "TEST")

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config
        }
        end = time.time()
        if epoch % 200 == 0 and epoch > 0 and myconfig["create_checkpoints"]:
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {
                    "train_L": train_loss_info["L"],
                    "train_LR": train_loss_info["LR"],
                    "train_LKL": train_loss_info["LKL"],
                    "test_LR": test_loss_info["LR"],
                    "coord err": test_debug_info["CE"], 
                    "num gauss": test_debug_info["num_gaussian"],
                    "eta_step": config["eta_step"],
                    "duration": end - start, 
                    "train": train_loss_info,
                    "test": test_loss_info},
                    checkpoint = checkpoint
                )
        else:
            train.report(
                    {
                    "train_L": train_loss_info["L"],
                    "train_LR": train_loss_info["LR"],
                    "train_LKL": train_loss_info["LKL"],
                    "test_L": test_loss_info["L"],
                    "coord err": test_debug_info["CE"], 
                    "num gauss": test_debug_info["num_gaussian"],
                    "eta_step": config["eta_step"],
                    "duration": end - start, 
                    "train": train_loss_info,
                    "test": test_loss_info},
                )


if __name__ == "__main__":

    myconfig = {
        "model": {
            "M": tune.grid_search([8, 16]),#tune.choice([1, 8, 16]),
            "enc_hidden_size": tune.grid_search([64, 128, 256]),
            "dec_hidden_size": None,#tune.choice([256, 512]),
            "Nz": None,
            "dropout": 0.1,#tune.choice([0.0, 0.1]),
            "deterministic": False,
        },
        "batch_size": 5000,#2000,
        "eta_min": 0.001,
        "eta_step": None,
        "R": 0.966,
        "KL_min": 0.2,#tune.choice([0.01, 0.25, 1]),
        "wKL": None,#tune.choice([0.1, 0.5, 1]),   #1
        "lr": 0.001,#tune.loguniform(1e-5, 1e-3),
        "lr_decay": 0.1,
        "lr_patience": 50,
        "lr_min": 0.00001,
        "grad_clip": 1,
        "temperature": 0.4,
        "max_seq_length": None,
        "epochs": 201,
        "image_interval": 1,
        "dataset": "fscoco",
        "coordinate_stye": "relative",
        "do_augment": True,
        "create_checkpoints": True,
        "num_samples": 1
    }
    assert myconfig["coordinate_stye"] == "relative" #FIXME: Not yet implemented

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Update configuration via command line.")

    # Dynamically add arguments based on the config dictionary
    recursive_add_arguments(parser, "", myconfig)

    # Parse the arguments
    args = parser.parse_args()

    # Update the config dictionary based on the command-line arguments
    recursive_update_config(myconfig, args)

    print("Config")
    print(myconfig)

    with tempfile.TemporaryDirectory() as dataset_dir:
        dataset_path = Path(dataset_dir) / "data.pkl"
        print("Started generating dataset")
        if myconfig["dataset"] == "fscoco":
            dataset = FSCOCOLoader(mode = "pretrain", max_length=myconfig["max_seq_length"], do_normalise = True, do_simplify = True)
        elif myconfig["dataset"] == "geometric":
            dataset = GeometrySet(mode = "pretrain", size = 10000, num_distribution = [0.1, 0.2, 0.3, 0.4], shape_distribution = [0.0, 0.3, 0.3, 0.4], noise = 5)
        print("Generated dataset with", len(dataset), "examples")
        torch.save(dataset, dataset_path)

        myconfig["dataset_path"] = dataset_path

        # scheduler = ASHAScheduler(
        #     metric="test/L",
        #     mode="min",
        #     max_t=myconfig["epochs"],
        #     grace_period=25,
        #     reduction_factor=2,
        # )

        # def custom_resources_per_trial(config):
        #     # Assign custom resource `encoder512_gpu` if encoder_size is 512
        #     if config["model"]["enc_hidden_size"] == 512:
        #         return {"cpu": 8, "gpu": 1, "encoder512_gpu": 1}
        #     else:
        #         return {"cpu": 8, "gpu": 1}

        result = tune.run(
            train_sketch_rnn,#partial(train_sketch_rnn, data_dir=data_dir),
            resources_per_trial={"cpu": 8, "gpu": 1/2},
            config=myconfig,
            num_samples=myconfig["num_samples"],
            # resources_per_trial=custom_resources_per_trial, #Don't put two large models on one gpu
            verbose = 1,
            # stop=EarlyStopper("test_LR", 20)
            # scheduler=scheduler,
            #stop=tune.stopper.TrialPlateauStopper("test/L", std = 0.001)
        )
