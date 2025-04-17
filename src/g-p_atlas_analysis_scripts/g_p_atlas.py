import pickle as pk
import sys
import time as tm
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import FeatureAblation
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data.dataset import Dataset
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

"""This is an implementation of the G-P Atlas method for mapping genotype
to phenotype described in https://doi.org/10.57844/arcadia-d316-721f.
For help type:
python3 g_p_atlas.py --help"""

# parse commandline arguments
args = ArgumentParser()
args.add_argument(
    "--n_alleles",
    type=int,
    default=2,
    help="number of segregating causal alleles at any given causal locus",
)
args.add_argument(
    "--n_locs",
    type=int,
    default=900,
    help="number of causal loci to model.  This is the same as the number of genes",
)
args.add_argument(
    "--n_env", type=int, default=3, help="number of influential continuous components"
)
args.add_argument("--n_phens", type=int, default=30, help="number of phenotypes")
args.add_argument(
    "--gen_lw", type=float, default=1, help="weight for the loss attributed to genetic features"
)
args.add_argument(
    "--eng_lw", type=float, default=0.1, help="weight for the loss attributed to env features"
)
args.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
args.add_argument("--batch_size", type=int, default=16, help="batch size")
args.add_argument("--lr_r", type=float, default=0.001, help="reconstruction learning rate")
args.add_argument("--b1", type=float, default=0.5, help="adam: gradient decay variables")
args.add_argument("--b2", type=float, default=0.999, help="adam: gradient decay variables")
args.add_argument("--n_cpu", type=int, default=14, help="number of cpus")
args.add_argument(
    "--e_hidden_dim", type=int, default=32, help="number of neurons in the hidden layers of encoder"
)
args.add_argument(
    "--d_hidden_dim", type=int, default=32, help="number of neurons in the hidden layers of decoder"
)
args.add_argument(
    "--ge_hidden_dim",
    type=int,
    default=32,
    help="number of neurons in the hidden layers of the genetic encoder",
)
args.add_argument(
    "--batchnorm_momentum",
    type=float,
    default=0.8,
    help="momentum for the batch normalization layers",
)
args.add_argument(
    "--latent_dim", type=int, default=32, help="number of neurons in the latent space"
)
args.add_argument(
    "--n_phens_to_analyze", type=int, default=30, help="number of phenotypes to analyze"
)
args.add_argument("--sd_noise", type=float, default=0.1, help="noise added to phens")
args.add_argument("--gen_noise", type=float, default=0.3, help="noise added to gens")
args.add_argument(
    "--n_phens_to_predict", type=int, default=30, help="number of phenotypes to predict"
)
args.add_argument(
    "--n_epochs_gen", type=int, default=100, help="number of epochs to train the genetic encoder"
)
args.add_argument(
    "--n_loci_measured",
    type=int,
    default=3000,
    help="number of genetic loci for which there is data",
)
args.add_argument(
    "--l1_lambda", type=float, default=0.8, help="l1 regularization weight for the genetic weights"
)
args.add_argument(
    "--l2_lambda",
    type=float,
    default=0.01,
    help="l2 regularization weight for the genetic weights",
)
args.add_argument(
    "--dataset_path", type=str, default=None, help="where the train and test files are located"
)
args.add_argument(
    "--train_suffix",
    type=str,
    default="train.pk",
    help="name of the training data file, defaults to DGRP dataset",
)
args.add_argument("--test_suffix", type=str, default="test.pk", help="name of the test data file")
args.add_argument(
    "--hot_start",
    type=bool,
    default=False,
    help="flag to use precomputed weights, false by default",
)
args.add_argument(
    "--hot_start_path_e",
    type=str,
    default=None,
    help="path to the phenotype encoder weights to use at initialization",
)
args.add_argument(
    "--hot_start_path_d",
    type=str,
    default=None,
    help="path to the phenotype decoder weights to use at initialization",
)
args.add_argument(
    "--hot_start_path_ge",
    type=str,
    default=None,
    help="path to the genotype encoder weights to use at initialization",
)
args.add_argument(
    "--calculate_importance",
    type=str,
    default="no",
    help="flag whether to calculate variable importance. Expects 'no' or 'yes.' ",
)
args.add_argument(
    "--detect_interactions",
    type=str,
    default="no",
    help="flag whether to detect gene-gene interactions. Expects 'no' or 'yes.' ",
)
args.add_argument(
    "--interaction_threshold",
    type=float,
    default=0.05,
    help="threshold for considering a gene-gene interaction significant",
)
args.add_argument(
    "--max_loci_interactions",
    type=int,
    default=100,
    help="maximum number of loci to test for interactions (to limit computation)",
)

vabs = args.parse_args()


# define a torch dataset object
class dataset_pheno(Dataset):
    """a class for importing simulated phenotype data.
    It expects a pickled object that is organized as a dictionary of tensors:
    phenotypes[n_animals,n_phens] float value for phenotype
    """

    def __init__(self, data_file, n_phens):
        self.datset = pk.load(open(data_file, "rb"))
        self.phens = torch.tensor(np.array(self.datset["phenotypes"]), dtype=torch.float32)
        self.data_file = data_file
        self.n_phens = n_phens

    def __len__(self):
        return len(self.phens)

    def __getitem__(self, idx):
        phenotypes = self.phens[idx][: self.n_phens]
        return phenotypes


class dataset_geno(Dataset):
    """a class for importing simulated genotype and phenotype data.
    It expects a pickled object that is organized as a dictionary of tensors:
    genotypes[n_animals, n_loci, n_alleles] (one hot at allelic state)
    phenotypes[n_animals,n_phens] float value for phenotype
    """

    def __init__(self, data_file, n_geno, n_phens):
        self.datset = pk.load(open(data_file, "rb"))
        self.phens = torch.tensor(np.array(self.datset["phenotypes"]), dtype=torch.float32)
        self.genotypes = torch.tensor(np.array(self.datset["genotypes"]), dtype=torch.float32)
        self.data_file = data_file
        self.n_geno = n_geno
        self.n_phens = n_phens

    def __len__(self):
        return len(self.genotypes)

    def __getitem__(self, idx):
        phenotypes = self.phens[idx][: self.n_phens]
        genotype = torch.flatten(self.genotypes[idx])
        return phenotypes, genotype


# helper functions
def sequential_forward_attr_gen_phen(input, phens):
    """puts together two models for use of captum feature
    importance in genotype-phenotype prediction"""
    mod_2_input = GQ(input)
    X_sample = P(mod_2_input)
    output = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)
    return output


def sequential_forward_attr_phen_phen(input, phens):
    """puts together two models for use of captum feature
    importance in phenotype-phenotype prediction"""
    mod_2_input = Q(input)
    X_sample = P(mod_2_input)
    output = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)
    return output


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true + EPS), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# load the training and test datasets
dataset_path = vabs.dataset_path

params_file = open(dataset_path + "run_params.txt", "w")
params_file.write(" ".join(sys.argv[:]))
params_file.close()

train_dat = vabs.train_suffix
test_dat = vabs.test_suffix

train_data_pheno = dataset_pheno(dataset_path + train_dat, n_phens=vabs.n_phens_to_analyze)
test_data_pheno = dataset_pheno(dataset_path + test_dat, n_phens=vabs.n_phens_to_analyze)

train_data_geno = dataset_geno(
    dataset_path + train_dat, n_geno=vabs.n_loci_measured, n_phens=vabs.n_phens_to_analyze
)
test_data_geno = dataset_geno(
    dataset_path + test_dat, n_geno=vabs.n_loci_measured, n_phens=vabs.n_phens_to_analyze
)

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# how many samples per batch to load
batch_size = vabs.batch_size
num_workers = vabs.n_cpu

# prepare data loaders
train_loader_pheno = torch.utils.data.DataLoader(
    dataset=train_data_pheno, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
test_loader_pheno = torch.utils.data.DataLoader(
    dataset=test_data_pheno, batch_size=1, num_workers=num_workers, shuffle=True
)

train_loader_geno = torch.utils.data.DataLoader(
    dataset=train_data_geno, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
test_loader_geno = torch.utils.data.DataLoader(
    dataset=test_data_geno, batch_size=1, num_workers=num_workers, shuffle=True
)


# model (encoders and decoders) classes
# phenotype encoder class
class Q_net(nn.Module):
    """
    Encoder for creating embeddings of phenotypic data.
    Parameters:
        out_phen_dim (int): Number of output phenotypes.
        N (int): Number of channels in hidden layers.
    """

    def __init__(self, phen_dim=None, N=None):
        super().__init__()
        if N is None:
            N = vabs.e_hidden_dim
        if phen_dim is None:
            phen_dim = vabs.n_phens_to_analyze

        batchnorm_momentum = vabs.batchnorm_momentum
        latent_dim = vabs.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features=phen_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(in_features=N, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# phenotype decoder class
class P_net(nn.Module):
    """
    Decoder for predicting phenotypes from either genotypic or phenotypic data.
    Parameters:
        out_phen_dim (int): Number of output phenotypes.
        N (int): Number of channels in hidden layers.
    """

    def __init__(self, out_phen_dim=None, N=None):
        if N is None:
            N = vabs.d_hidden_dim
        if out_phen_dim is None:
            out_phen_dim = vabs.n_phens_to_predict

        latent_dim = vabs.latent_dim
        batchnorm_momentum = vabs.batchnorm_momentum

        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=out_phen_dim),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


# genetic encoder class
class GQ_net(nn.Module):
    """
    Genetic encoder to produce latent embedding of genotypic data for predicting
    either phenotypes or genotypes.

    Parameters:
        n_loci (int): number of input measured loci * number of segregating alleles
        N (int): Number of channels in hidden layers.
    """

    def __init__(self, n_loci=None, N=None):
        super().__init__()
        if N is None:
            N = vabs.ge_hidden_dim
        if n_loci is None:
            n_loci = vabs.n_loci_measured * vabs.n_alleles

        batchnorm_momentum = vabs.batchnorm_momentum
        latent_dim = vabs.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features=n_loci, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# G-P Atlas run

EPS = 1e-15  # define a minimum value for variables to avoid divide by zero and buffer underflow

# define encoders and decoders
Q = Q_net()  # phenotype encoder
P = P_net()  # phenotype decoder
GQ = GQ_net()  # genotype encoder

# load precomputed weights
# this allows you to specify precomputed weights so that the training can be 'hotstarted'
if vabs.hot_start is True:
    Q.load_state_dict(torch.load(vabs.hot_start_path_e, weights_only=True), strict=False)
    P.load_state_dict(torch.load(vabs.hot_start_path_d, weights_only=True), strict=False)
    GQ.load_state_dict(torch.load(vabs.hot_start_path_ge, weights_only=True), strict=False)

# put the models on the GPU if it is there
Q.to(device)
P.to(device)
GQ.to(device)

# Set up feature importance measure
fa = FeatureAblation(sequential_forward_attr_gen_phen)  # genotype feature importance
fa_p = FeatureAblation(sequential_forward_attr_phen_phen)  # phenotype feature importance

# Set learning rates
reg_lr = vabs.lr_r

# adam betas
adam_b = (vabs.b1, vabs.b2)

# encode/decode optimizers
optim_P = torch.optim.Adam(P.parameters(), lr=reg_lr, betas=adam_b)
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=reg_lr, betas=adam_b)
optim_GQ_enc = torch.optim.Adam(GQ.parameters(), lr=reg_lr, betas=adam_b)

# set the number of training epochs for the phenotype-phenotype autoencoder
num_epochs = vabs.n_epochs

torch.manual_seed(47)

# establish the number of phenotypes to predict and the number to use for prediction.
n_phens = vabs.n_phens_to_analyze
n_phens_pred = vabs.n_phens_to_predict

# establish a variable to capture the reconstruction loss
rcon_loss = []

# establish a variable for the start of the run
start_time = tm.time()

# train the phenotype encoder and decoder
for n in range(num_epochs):
    for _i, (phens) in enumerate(train_loader_pheno):
        phens = phens[:, :n_phens]  # constrain the number of phenotypes to use for prediction
        phens = phens.to(device)  # move data to GPU if it is there
        batch_size = phens.shape[0]  # redefine batch size here to allow for incomplete batches

        P.zero_grad()  # initialize gradients for training
        Q.zero_grad()

        noise_phens = phens + (vabs.sd_noise**0.5) * torch.randn(phens.shape).to(device)
        # add noise to phenotypes

        z_sample = Q(noise_phens)  # encode phenotypes
        X_sample = P(z_sample)  # decode encodings to produce predicted phenotypes

        recon_loss = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)
        # calculate the error of the phenotype predicitons

        rcon_loss.append(float(recon_loss.detach()))  # add the loss to the aggregator

        recon_loss.backward()  # back propagate the reconstruction loss through the autoencoder
        optim_P.step()  # step the optimizers
        optim_Q_enc.step()

    cur_time = tm.time() - start_time  # calculate the time it took for this batch
    start_time = tm.time()  # re-initialize the start time
    # for each loop, print a set of useful information
    print(
        "Epoch num: "
        + str(n)
        + " batchno "
        + str(_i)
        + " r_con_loss: "
        + str(rcon_loss[-1])
        + " epoch duration: "
        + str(cur_time)
    )


# train genetic network

P.requires_grad_(False)  # freeze weights in P (decoder)
P.eval()  # put P (phenotype decoder) into evaluation mode
num_epochs_gen = vabs.n_epochs_gen

gen_noise = 1 - vabs.gen_noise

g_rcon_loss = []  # establish a variable to contain the reconstruction loss values

start_time = tm.time()

for n in range(num_epochs_gen):
    for _i, (phens, gens) in enumerate(train_loader_geno):
        phens = phens.to(device)  # move phenotypic data to the gpu if it is there

        gens = gens[:, : vabs.n_loci_measured * vabs.n_alleles]

        pos_noise = np.random.binomial(1, gen_noise / 2, gens.shape)

        neg_noise = np.random.binomial(1, gen_noise / 2, gens.shape)

        noise_gens = torch.tensor(
            np.where((gens + pos_noise - neg_noise) > 0, 1, 0), dtype=torch.float32
        )  # add noise to the genetic data

        noise_gens = noise_gens.to(device)  # put genotypes plus noise on the gpu if it is there

        batch_size = phens.shape[0]  # establish the training batch size

        GQ.zero_grad()  # zero the gradients

        z_sample = GQ(noise_gens)  # encode the genetic data
        X_sample = P(z_sample)  # decode the encoded genetic data to phenotypes

        g_recon_loss = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)

        g_rcon_loss.append(float(g_recon_loss.detach()))

        # Calculate the L1 and L2 norms for the weights in the first layer of the
        # genetic encoder and add them to the reconstruction loss
        l1_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 1)
        l2_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 2)
        g_recon_loss = g_recon_loss + l1_reg * vabs.l1_lambda + l2_reg * vabs.l2_lambda

        g_recon_loss.backward()  # backpropagate the loss

        optim_GQ_enc.step()  # step the optimizer

    cur_time = tm.time() - start_time  # set a variable for the epoch time
    start_time = tm.time()  # set the original start time variable to the current time

    # print useful things about the current training epoch
    print(
        "Epoch num: "
        + str(n)
        + " batchno "
        + str(_i)
        + " r_con_loss: "
        + str(g_rcon_loss[-1])
        + " epoch duration: "
        + str(cur_time)
    )

# plot the reconstruction losses
plt.plot(rcon_loss)  # reconstruction loss for the phenotype autoencoder
plt.plot(g_rcon_loss)  # reconstruction loss for the genetic weights
plt.savefig(dataset_path + "reconstruction_loss.svg")
plt.close()


# A function to evaluate the performance of each model, saving summaries of model performance
def analyze_predictions(
    phens,
    phen_encodings,
    phen_latent,
    fa_attr,
    dataset_path,
    n_phens_pred,
    model_type="g_p",
):
    """Analyze predictions and save visualization results.

    Parameters
    ----------
    phens : list
        List of real phenotype values (numpy arrays) from model evaluation
    phen_encodings : list
        List of predicted phenotype values (numpy arrays) from model evaluation
    phen_latent : list
        List of latent space representations (numpy arrays) from model evaluation
    fa_attr : list
        List of feature attribution values (numpy arrays) from captum analysis
    dataset_path : str
        Path to directory where all output files should be saved
    n_phens_pred : int
        Number of phenotypes to analyze and plot (uses first n_phens_pred phenotypes)
    model_type : str, optional
        Type of prediction being analyzed - either 'g_p' (genotype-phenotype) or
        'p_p' (phenotype-phenotype). Affects file naming and data saving. Default is 'g_p'

    Returns
    -------
    list
        List of metric results, containing in order: Pearson correlations, MSE values,
        MAPE values, and R2 scores for the first n_phens_pred phenotypes
    """
    suffix = "_p" if model_type == "p_p" else ""

    # Save attributions
    plt.hist(fa_attr, bins=20)
    plt.savefig(dataset_path + f"{model_type}_attr.svg")
    plt.close()
    if fa_attr != []:
        pk.dump(fa_attr, open(dataset_path + f"{model_type}_attr.pk", "wb"))

    # Convert and transpose data
    phens = np.array(phens).T
    phen_encodings = np.array(phen_encodings).T

    # Save predictions data
    if model_type == "p_p":
        phen_latent = np.array(phen_latent).T
        pk.dump(
            [phens, phen_encodings, phen_latent],
            open(dataset_path + f"phens_phen_encodings_dng_attr{suffix}.pk", "wb"),
        )
    else:
        pk.dump(
            [phens, phen_encodings],
            open(dataset_path + f"phens_phen_encodings_dng_attr{suffix}.pk", "wb"),
        )

    # Plot predictions
    for n in range(len(phens[:n_phens_pred])):
        plt.plot(phens[n], phen_encodings[n], "o")
    plt.xlabel("real")
    plt.ylabel("predicted")
    plt.gca().set_aspect("equal")
    plt.savefig(dataset_path + f"phen_real_pred_dng_attr{suffix}.svg")
    plt.close()

    # Calculate and plot metrics
    stats_aggregator = []

    # Pearson correlation
    cors = [
        sc.stats.pearsonr(phens[n], phen_encodings[n])[0] for n in range(len(phens[:n_phens_pred]))
    ]
    stats_aggregator.append(cors)
    plt.hist(cors, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_pearsonsr_dng_attr{suffix}.svg")
    plt.close()

    # MSE
    errs = [
        mean_squared_error(phens[n], phen_encodings[n]) for n in range(len(phens[:n_phens_pred]))
    ]
    stats_aggregator.append(errs)
    plt.hist(errs, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_mse_dng_attr{suffix}.svg")
    plt.close()

    # MAPE
    errs = [
        mean_absolute_percentage_error(phens[n], phen_encodings[n])
        for n in range(len(phens[:n_phens_pred]))
    ]
    stats_aggregator.append(errs)
    plt.hist(errs, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_mape_dng_attr{suffix}.svg")
    plt.close()

    # R2
    errs = [r2_score(phens[n], phen_encodings[n]) for n in range(len(phens[:n_phens_pred]))]
    stats_aggregator.append(errs)
    plt.hist(errs, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_r2_dng_attr{suffix}.svg")
    plt.close()

    return stats_aggregator


# Now use this function to analyze predictions and save outputs
out_stats = open(dataset_path + "test_stats.pk", "wb")
stats_aggregator = []

# Save model states first
torch.save(Q.state_dict(), dataset_path + "phen_encoder_state.pt")
torch.save(P.state_dict(), dataset_path + "phen_decoder_state.pt")
torch.save(GQ.state_dict(), dataset_path + "gen_encoder_state.pt")

# G-P prediction. Loop evaluating the performance on the test data
GQ.eval()
phens, phen_encodings, phen_latent, fa_attr = [], [], [], []

for dat in test_loader_geno:
    ph, gt = dat
    gt = gt[:, : vabs.n_loci_measured * vabs.n_alleles]
    ph = ph.to(device)
    gt = gt.to(device)
    batch_size = ph.shape[0]
    z_sample = GQ(gt)
    X_sample = P(z_sample)
    phens += list(ph.detach().cpu().numpy())
    phen_encodings += list(X_sample.detach().cpu().numpy())
    phen_latent += list(z_sample.detach().cpu().numpy())
    if vabs.calculate_importance == "yes":
        fa_attr.append(list(fa.attribute(inputs=(gt, ph))[0].squeeze().detach().cpu().numpy()))

stats_aggregator.extend(
    analyze_predictions(
        phens, phen_encodings, phen_latent, fa_attr, dataset_path, n_phens_pred, "g_p"
    )
)

# P-P prediction. Loop evaluating the performance on the test data
Q.eval()
phens, phen_encodings, phen_latent, fa_attr = [], [], [], []

for dat in test_loader_pheno:
    ph = dat
    ph = ph.to(device)
    batch_size = ph.shape[0]
    z_sample = Q(ph)
    X_sample = P(z_sample)
    phens += list(ph.detach().cpu().numpy())
    phen_encodings += list(X_sample.detach().cpu().numpy())
    phen_latent += list(z_sample.detach().cpu().numpy())
    if vabs.calculate_importance == "yes":
        fa_attr.append(list(fa_p.attribute(inputs=(ph, ph))[0].squeeze().detach().cpu().numpy()))

stats_aggregator.extend(
    analyze_predictions(
        phens, phen_encodings, phen_latent, fa_attr, dataset_path, n_phens_pred, "p_p"
    )
)

# Detection of gene-gene interactions
if vabs.detect_interactions == "yes":
    print("Detecting gene-gene interactions...")
    
    # Use the comprehensive analysis function
    analyze_gene_interaction_network(GQ, P, test_loader_geno, dataset_path)

# Save and close stats
pk.dump(stats_aggregator, out_stats)
out_stats.close()

def analyze_gene_interaction_network(model, P, test_loader, dataset_path):
    """
    Perform final analysis of gene-gene interactions and save results.
    
    Parameters:
    -----------
    model : GQ_net
        Trained genotype encoder model
    P : P_net
        Trained phenotype decoder model
    test_loader : DataLoader
        Loader for test dataset
    dataset_path : str
        Path to save results
    """
    # Collect all test data
    all_genotypes = []
    all_phenotypes = []
    
    for dat in test_loader:
        ph, gt = dat
        gt = gt[:, : vabs.n_loci_measured * vabs.n_alleles]
        all_genotypes.append(gt)
        all_phenotypes.append(ph)
    
    all_genotypes = torch.cat(all_genotypes, dim=0)
    all_phenotypes = torch.cat(all_phenotypes, dim=0)
    
    # Find interactions for each phenotype
    n_phenotypes = min(3, all_phenotypes.shape[1])  # Limit to first 3 phenotypes for computation
    all_interaction_data = {}
    
    for phen_idx in range(n_phenotypes):
        print(f"Analyzing interactions for phenotype {phen_idx+1}/{n_phenotypes}")
        
        # Calculate interactions for this phenotype
        interactions = detect_pairwise_interactions(
            model, P, all_genotypes, all_phenotypes, 
            phenotype_idx=phen_idx, threshold=vabs.interaction_threshold)
        
        # Convert to dictionary format for compatibility
        interaction_dict = {(i, j): strength for i, j, strength in interactions}
        
        # Plot network
        plt.figure(figsize=(12, 10))
        fig = plot_interaction_network(
            interactions, phenotype_name=f"Phenotype {phen_idx}", threshold=vabs.interaction_threshold)
        
        # Save plot
        plt.savefig(dataset_path + f"interaction_network_phenotype_{phen_idx}.svg")
        plt.savefig(dataset_path + f"interaction_network_phenotype_{phen_idx}.png")
        plt.close()
        
        # Store data
        all_interaction_data[phen_idx] = interaction_dict
    
    # Save all interaction data
    with open(dataset_path + "gene_gene_interactions.pk", "wb") as f:
        pk.dump(all_interaction_data, f)
    
    # Create a summary of the most significant interactions across phenotypes
    interaction_summary = {}
    
    for phen_idx, interactions in all_interaction_data.items():
        for (i, j), importance in interactions.items():
            if (i, j) not in interaction_summary:
                interaction_summary[(i, j)] = []
            interaction_summary[(i, j)].append((phen_idx, importance))
    
    # Find interactions affecting multiple phenotypes
    multi_phenotype_interactions = {k: v for k, v in interaction_summary.items() if len(v) > 1}
    
    # Sort by total importance across phenotypes
    sorted_multi = sorted(multi_phenotype_interactions.items(), 
                        key=lambda x: sum(imp for _, imp in x[1]), 
                        reverse=True)
    
    # Save top multi-phenotype interactions
    with open(dataset_path + "multi_phenotype_interactions.pk", "wb") as f:
        pk.dump(sorted_multi, f)
    
    print(f"Saved interaction analysis results to {dataset_path}")
    
    return all_interaction_data

def detect_pairwise_interactions(model, P, genotypes, phenotypes, phenotype_idx=0, threshold=0.05):
    """
    Detects pairwise interactions between genetic loci by measuring the non-additive 
    effects on phenotype predictions.
    
    This method works by calculating the difference between the expected additive effect
    of modifying two loci independently and the actual effect of modifying them together.
    A non-zero difference indicates a gene-gene interaction (epistasis).
    
    Parameters:
    -----------
    model : GQ_net
        Trained genotype encoder model
    P : P_net
        Trained phenotype decoder model
    genotypes : torch.Tensor
        Genotype data
    phenotypes : torch.Tensor
        Ground truth phenotype data
    phenotype_idx : int
        Index of the phenotype to analyze
    threshold : float
        Threshold for considering an interaction significant
        
    Returns:
    --------
    interactions : list of tuples
        List of (locus1, locus2, interaction_strength) tuples
    """
    n_loci = genotypes.shape[1] // vabs.n_alleles
    
    # Move models to evaluation mode
    model.eval()
    P.eval()
    
    # Store the device
    device = next(model.parameters()).device
    
    # Initialize interactions list
    interactions = []
    
    # Limit the number of loci to test for interactions if needed
    max_loci = min(n_loci, vabs.max_loci_interactions)
    if max_loci < n_loci:
        print(f"Testing interactions for first {max_loci} loci out of {n_loci} total")
    
    # For each pair of loci
    for i in range(max_loci - 1):
        for j in range(i + 1, max_loci):
            # Get the indices for the alleles at these loci
            i_start = i * vabs.n_alleles
            i_end = i_start + vabs.n_alleles
            j_start = j * vabs.n_alleles
            j_end = j_start + vabs.n_alleles
            
            # Create modified genotypes for measuring individual effects
            genotypes_i_modified = genotypes.clone()
            genotypes_i_modified[:, i_start:i_end] = 0.5  # Set to average value
            
            genotypes_j_modified = genotypes.clone()
            genotypes_j_modified[:, j_start:j_end] = 0.5  # Set to average value
            
            genotypes_both_modified = genotypes.clone()
            genotypes_both_modified[:, i_start:i_end] = 0.5
            genotypes_both_modified[:, j_start:j_end] = 0.5
            
            # Get predictions
            with torch.no_grad():
                pred_baseline = P(model(genotypes.to(device)))[:, phenotype_idx]
                pred_i_modified = P(model(genotypes_i_modified.to(device)))[:, phenotype_idx]
                pred_j_modified = P(model(genotypes_j_modified.to(device)))[:, phenotype_idx]
                pred_both_modified = P(model(genotypes_both_modified.to(device)))[:, phenotype_idx]
            
            # Calculate effects
            effect_i = pred_baseline - pred_i_modified
            effect_j = pred_baseline - pred_j_modified
            
            # Expected additive effect
            expected_both_effect = effect_i + effect_j
            
            # Actual effect
            actual_both_effect = pred_baseline - pred_both_modified
            
            # Interaction strength - the difference between actual and expected effects
            # Normalized by the standard deviation of phenotype values for better comparability
            pheno_std = phenotypes[:, phenotype_idx].std().item()
            interaction_strength = torch.abs(actual_both_effect - expected_both_effect).mean().item()
            normalized_strength = interaction_strength / (pheno_std + EPS)
            
            # If interaction is significant, add to list
            if normalized_strength > threshold:
                interactions.append((i, j, normalized_strength))
    
    # Sort by interaction strength
    interactions.sort(key=lambda x: x[2], reverse=True)
    
    return interactions

def plot_interaction_network(interaction_importance, locus_names=None, phenotype_name=None,
                             threshold=0.05, max_nodes=50):
    """
    Visualize gene-gene interactions as a network.
    
    Parameters:
    -----------
    interaction_importance : list of tuples
        List of (locus1, locus2, interaction_strength) tuples
    locus_names : list, optional
        List of names for each locus
    phenotype_name : str, optional
        Name of the phenotype being analyzed
    threshold : float
        Minimum importance score to include in the visualization
    max_nodes : int
        Maximum number of nodes to include in the visualization
    """
    # Filter interactions by threshold
    filtered_interactions = [(i, j, strength) for i, j, strength in interaction_importance if strength > threshold]
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes and edges
    all_loci = set()
    for i, j, importance in filtered_interactions:
        all_loci.add(i)
        all_loci.add(j)
    
    # Limit number of nodes if necessary
    if len(all_loci) > max_nodes:
        # Find the max_nodes most important loci
        locus_importance = {}
        for i, j, importance in filtered_interactions:
            locus_importance[i] = locus_importance.get(i, 0) + importance
            locus_importance[j] = locus_importance.get(j, 0) + importance
        
        top_loci = sorted(locus_importance.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_loci = set(x[0] for x in top_loci)
        filtered_interactions = [(i, j, v) for i, j, v in filtered_interactions 
                                if i in top_loci and j in top_loci]
    
    # Add nodes
    for i, j, importance in filtered_interactions:
        if locus_names is not None:
            G.add_node(i, name=locus_names[i])
            G.add_node(j, name=locus_names[j])
        else:
            G.add_node(i, name=f'Locus {i}')
            G.add_node(j, name=f'Locus {j}')
        G.add_edge(i, j, weight=importance)
    
    # Set up plot
    plt.figure(figsize=(12, 12))
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get edge weights for colors
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    if not edge_weights:
        plt.title(f"No significant interactions found for {phenotype_name}")
        return plt.gcf()
    
    # Normalize weights for colormap
    norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8)
    
    # Draw edges with color based on weight
    edges = nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_weights, 
                                  edge_cmap=plt.cm.viridis, edge_vmin=min(edge_weights), 
                                  edge_vmax=max(edge_weights))
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['name'] for n in G.nodes()}, 
                            font_size=10, font_family='sans-serif')
    
    # Add colorbar
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Interaction Importance')
    
    # Set title
    if phenotype_name is not None:
        plt.title(f'Gene-Gene Interaction Network for {phenotype_name}', fontsize=16)
    else:
        plt.title('Gene-Gene Interaction Network', fontsize=16)
    
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()
