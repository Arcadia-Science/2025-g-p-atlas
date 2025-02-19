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
from torchvision import transforms

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
    help="momentum for the batchnormalization layers",
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
    "--l1_lambda", type=float, default=0.8, help="l1 regularlization weight for the genetic weights"
)
args.add_argument(
    "--l2_lambda",
    type=float,
    default=0.01,
    help="l2 regularlization weight for the genetic weights",
)
args.add_argument(
    "--dataset_path", type=str, default=None, help="where the train and test files are located"
)
args.add_argument(
    "--train_suffix",
    type=str,
    default="dgrp_g_p_train_set.pk",
    help="name of the training data file",
)
args.add_argument(
    "--test_suffix", type=str, default="dgrp_g_p_test_set.pk", help="name of the test data file"
)
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


vabs = args.parse_args()


# define a torch dataset object
class dataset_dgrp_pheno(Dataset):
    """a class for importing simulated genotype-phenotype data.
    It expects a pickled object that is organized as a list of tensors:
    genotypes[n_animals, n_loci, n_alleles] (one hot at allelic state)
    gen_locs[n_animals, n_loci] (index of allelic state)
    weights[n_phens, n_loci, n_alleles] float weight for allelic contribution to phen
    phens[n_animals,n_phens] float value for phenotype
    indexes_of_loci_influencing_phen[n_phens,n_loci_ip] integer indicies of
    loci that influence a phenotype interaction_matrix
    pleiotropy_matrix[n_phens, n_phens, gen_index]"""

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


class dataset_dgrp_geno(Dataset):
    """a class for importing simulated genotype-phenotype data.
    It expects a pickled object that is organized as a list of tensors:
    genotypes[n_animals, n_loci, n_alleles] (one hot at allelic state)
    gen_locs[n_animals, n_loci] (index of allelic state)
    weights[n_phens, n_loci, n_alleles] float weight for allelic contribution to phen
    phens[n_animals,n_phens] float value for phenotype
    indexes_of_loci_influencing_phen[n_phens,n_loci_ip] integer indicies of loci that
    influence a phenotype interaction_matrix[FILL THIS IN]
    pleiotropy_matrix[n_phens, n_phens, gen_index]"""

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
    """puts together two models for use of captum feature importance"""
    mod_2_input = GQ(input)
    X_sample = P(mod_2_input)
    output = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)
    return output


def sequential_forward_attr_phen_phen(input, phens):
    """puts together two models for use of captum feature importance"""
    mod_2_input = Q(input)
    X_sample = P(mod_2_input)
    output = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)
    return output


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true + EPS), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
dataset_path = vabs.dataset_path

params_file = open(dataset_path + "run_params.txt", "w")
params_file.write(" ".join(sys.argv[:]))
params_file.close()

train_dat = vabs.train_suffix
test_dat = vabs.test_suffix

train_data_pheno = dataset_dgrp_pheno(dataset_path + train_dat, n_phens=vabs.n_phens_to_analyze)
test_data_pheno = dataset_dgrp_pheno(dataset_path + test_dat, n_phens=vabs.n_phens_to_analyze)

train_data_geno = dataset_dgrp_geno(
    dataset_path + train_dat, n_geno=vabs.n_loci_measured, n_phens=vabs.n_phens_to_analyze
)
test_data_geno = dataset_dgrp_geno(
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
# encoder
class Q_net(nn.Module):
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


# decoder
class P_net(nn.Module):
    def __init__(self, phen_dim=None, N=None):
        if N is None:
            N = vabs.d_hidden_dim
        if phen_dim is None:
            phen_dim = vabs.n_phens_to_analyze

        out_phen_dim = vabs.n_phens_to_predict
        vabs.n_locs * vabs.n_alleles
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


# gencoder
class GQ_net(nn.Module):
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
P = P_net()  # phenotype deocoder
GQ = GQ_net()  # genotype encoder

# load precomputed weights
# this allows you to specify precomuted weights so that the training can be 'hotstarted'
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

num_epochs = vabs.n_epochs

torch.manual_seed(47)

# train phenotype autoencoder
n_phens = vabs.n_phens_to_analyze
n_phens_pred = vabs.n_phens_to_predict
rcon_loss = []

start_time = tm.time()

for n in range(num_epochs):
    for _i, (phens) in enumerate(train_loader_pheno):
        phens = phens[:, :n_phens]
        phens = phens.to(device)  # move data to GPU if it is there
        batch_size = phens.shape[0]  # redefine batch size here to allow for incomplete batches

        P.zero_grad()
        Q.zero_grad()

        noise_phens = phens + (vabs.sd_noise**0.5) * torch.randn(phens.shape).to(device)

        z_sample = Q(noise_phens)
        X_sample = P(z_sample)

        recon_loss = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)

        rcon_loss.append(float(recon_loss.detach()))

        recon_loss.backward()
        optim_P.step()
        optim_Q_enc.step()

    cur_time = tm.time() - start_time
    start_time = tm.time()
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
P.eval()
num_epochs_gen = vabs.n_epochs_gen

gen_noise = 1 - vabs.gen_noise

g_rcon_loss = []

start_time = tm.time()

for n in range(num_epochs_gen):
    for _i, (phens, gens) in enumerate(train_loader_geno):
        phens = phens.to(device)

        gens = gens[:, : vabs.n_loci_measured * vabs.n_alleles]

        pos_noise = np.random.binomial(1, gen_noise / 2, gens.shape)

        neg_noise = np.random.binomial(1, gen_noise / 2, gens.shape)

        noise_gens = torch.tensor(
            np.where((gens + pos_noise - neg_noise) > 0, 1, 0), dtype=torch.float32
        )

        noise_gens = noise_gens.to(device)

        batch_size = phens.shape[0]

        GQ.zero_grad()

        z_sample = GQ(noise_gens)
        X_sample = P(z_sample)

        g_recon_loss = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)

        g_rcon_loss.append(float(g_recon_loss.detach()))

        l1_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 1)
        l2_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 2)
        g_recon_loss = g_recon_loss + l1_reg * vabs.l1_lambda + l2_reg * vabs.l2_lambda

        g_recon_loss.backward()

        optim_GQ_enc.step()

    cur_time = tm.time() - start_time
    start_time = tm.time()
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

# test the g-p prediction
phen_encodings = []
phens = []
phen_latent = []
fa_attr = []

GQ.eval()

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
    fa_attr.append(list(fa.attribute(inputs=(gt, ph))[0].squeeze().detach().cpu().numpy()))


# plot a histogram of the genetic feature attributions
plt.hist(fa_attr, bins=20)
plt.savefig(dataset_path + "g_p_attr.svg")
plt.close()

# save the genetic feature attributions
pk.dump(fa_attr, open(dataset_path + "g_p_attr.pk", "wb"))

# save weights from networks
torch.save(Q.state_dict(), dataset_path + "phen_encoder_state.pt")
torch.save(P.state_dict(), dataset_path + "phen_decoder_state.pt")
torch.save(GQ.state_dict(), dataset_path + "gen_encoder_state.pt")

# save and plot phenotypes and phenotype predictions based on genotypes
phens = np.array(phens).T
phen_encodings = np.array(phen_encodings).T
pk.dump([phens, phen_encodings], open(dataset_path + "phens_phen_encodings_dng_attr.pk", "wb"))

# create a file and an agregator for statistics
out_stats = open(dataset_path + "test_stats.pk", "wb")
stats_agregator = []

for n in range(len(phens[:n_phens_pred])):
    plt.plot(phens[n], phen_encodings[n], "o")
plt.xlabel("real")
plt.ylabel("predicted")
plt.gca().set_aspect("equal")
plt.savefig(dataset_path + "phen_real_pred_dng_attr.svg")
plt.close()

cors = [sc.stats.pearsonr(phens[n], phen_encodings[n])[0] for n in range(len(phens[:n_phens_pred]))]
print(cors)
stats_agregator.append(cors)
plt.hist(cors, bins=20)
plt.savefig(dataset_path + "phen_real_pred_pearsonsr_dng_attr.svg")
plt.close()

errs = [mean_squared_error(phens[n], phen_encodings[n]) for n in range(len(phens[:n_phens_pred]))]
print(errs)
stats_agregator.append(errs)
plt.hist(errs, bins=20)
plt.savefig(dataset_path + "phen_real_pred_mse_dng_attr.svg")
plt.close()

errs = [
    mean_absolute_percentage_error(phens[n], phen_encodings[n])
    for n in range(len(phens[:n_phens_pred]))
]
print(errs)
stats_agregator.append(errs)
plt.hist(errs, bins=20)
plt.savefig(dataset_path + "phen_real_pred_mape_dng_attr.svg")
plt.close()

errs = [r2_score(phens[n], phen_encodings[n]) for n in range(len(phens[:n_phens_pred]))]
print(errs)
stats_agregator.append(errs)
plt.hist(errs, bins=20)
plt.savefig(dataset_path + "phen_real_pred_r2_dng_attr.svg")
plt.close()

# test the p-p prediction
Q.eval()

phen_encodings = []
phens = []
phen_latent = []
fa_attr = []

for dat in test_loader_pheno:
    ph = dat
    ph = ph.to(device)
    batch_size = ph.shape[0]
    z_sample = Q(ph)
    X_sample = P(z_sample)
    phens += list(ph.detach().cpu().numpy())
    phen_encodings += list(X_sample.detach().cpu().numpy())
    phen_latent += list(z_sample.detach().cpu().numpy())
    fa_attr.append(list(fa_p.attribute(inputs=(ph, ph))[0].squeeze().detach().cpu().numpy()))


plt.hist(fa_attr, bins=20)
plt.savefig(dataset_path + "p_p_attr.svg")
plt.close()
pk.dump(fa_attr, open(dataset_path + "p_p_attr.pk", "wb"))

phens = np.array(phens).T
phen_encodings = np.array(phen_encodings).T
phen_latent = np.array(phen_latent).T

pk.dump(
    [phens, phen_encodings, phen_latent],
    open(dataset_path + "phens_phen_encodings_dng_attr_p.pk", "wb"),
)


for n in range(len(phens[:n_phens_pred])):
    plt.plot(phens[n], phen_encodings[n], "o")
plt.xlabel("real")
plt.ylabel("predicted")
plt.gca().set_aspect("equal")
plt.savefig(dataset_path + "phen_real_pred_dng_attr_p.svg")
plt.close()

cors = [sc.stats.pearsonr(phens[n], phen_encodings[n])[0] for n in range(len(phens[:n_phens_pred]))]
print(cors)
stats_agregator.append(cors)
plt.hist(cors, bins=20)
plt.savefig(dataset_path + "phen_real_pred_pearsonsr_dng_attr_p.svg")
plt.close()

errs = [mean_squared_error(phens[n], phen_encodings[n]) for n in range(len(phens[:n_phens_pred]))]
print(errs)
stats_agregator.append(errs)
plt.hist(errs, bins=20)
plt.savefig(dataset_path + "phen_real_pred_mse_dng_attr_p.svg")
plt.close()

errs = [
    mean_absolute_percentage_error(phens[n], phen_encodings[n])
    for n in range(len(phens[:n_phens_pred]))
]
print(errs)
stats_agregator.append(errs)
plt.hist(errs, bins=20)
plt.savefig(dataset_path + "phen_real_pred_mape_dng_attr_p.svg")
plt.close()

errs = [r2_score(phens[n], phen_encodings[n]) for n in range(len(phens[:n_phens_pred]))]
print(errs)
stats_agregator.append(errs)
plt.hist(errs, bins=20)
plt.savefig(dataset_path + "phen_real_pred_r2_dng_attr_p.svg")
plt.close()

# save stats
pk.dump(stats_agregator, out_stats)
