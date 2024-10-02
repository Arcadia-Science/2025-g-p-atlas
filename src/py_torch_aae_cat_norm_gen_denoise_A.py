import pickle as pk
import time as tm
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision import transforms

#parse commandline arguments
args=ArgumentParser()
args.add_argument('--n_alleles', type=int, default=3, help='number of segregating causal alleles at any given causal locus')
args.add_argument('--n_locs', type=int, default=900, help='number of causal loci to model.  This is the same as the number of genes in the latent space')
args.add_argument('--n_env', type=int, default=3, help='number of influential continuous components')
args.add_argument('--n_phens', type=int, default=30, help='number of phenotypes')
args.add_argument('--gen_lw', type=float, default=1, help='weight for the loss attributed to genetic features')
args.add_argument('--eng_lw', type=float, default=0.1, help='weight for the loss attributed to env features')
args.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
args.add_argument('--batch_size', type=int, default=64, help='batch size')
args.add_argument('--lr_g', type=float, default=0.001, help='generator learning rate')
args.add_argument('--lr_r', type=float, default=0.00001, help='reconstruction learning rate')
args.add_argument('--lr_r_g', type=float, default=0.001, help='d_gauss learning rate')
args.add_argument('--lr_r_c', type=float, default=0.001, help='d_cat learning rate')
args.add_argument('--b1', type=float, default=0.5, help='adam: gradient decay variables')
args.add_argument('--b2', type=float, default=0.999, help='adam: gradient decay variables')
args.add_argument('--n_cpu', type=int, default=14, help='number of cpus')
args.add_argument('--e_hidden_dim',type=int,default=128, help='number of neurons in the hidden layers of encoder')
args.add_argument('--d_hidden_dim',type=int,default=128, help='number of neurons in the hidden layers of decoder')
args.add_argument('--batchnorm_momentum',type=float, default=0.8, help='momentum for the batchnormalization layers')
args.add_argument('--hidden_dim_dg', type=int, default=128, help='number of neurons in the hidden layers of the genetic descriminators')
args.add_argument('--hidden_dim_de', type=int, default=30, help='number of neurons in the hidden layers of the descriminators')

args.add_argument('--latent_dim', type=int, default=32, help='number of neurons in the categorical latent space')
args.add_argument('--n_phens_to_analyze', type=int, default=30, help='number of phenotypes to analyze')
args.add_argument('--n_phens_to_predict', type=int, default=30, help='number of phenotypes to predict')

args.add_argument('--sd_noise', type= float, default=0.1, help='noise added to phens')



vabs=args.parse_args()


#define a torch dataset object
class phen_dataset(Dataset):
 '''a class for importing simulated genotype-phenotype data.
 It expects a pickled object that is organized as a list of tensors:
 genotypes[n_animals, n_loci, n_alleles] (one hot at allelic state)
 gen_locs[n_animals, n_loci] (index of allelic state)
 weights[n_phens, n_loci, n_alleles] float weight for allelic contribution to phen
 phens[n_animals,n_phens] float value for phenotype
 indexes_of_loci_influencing_phen[n_phens,n_loci_ip] integer indicies of loci that influence a phenotype
 interaction_matrix[FILL THIS IN]
 pleiotropy_matrix[n_phens, n_phens, gen_index]'''
 def __init__(self,data_file,n_phens):
  self.datset = pk.load(open(data_file,'rb'))
  #self.phens = [list((x/(1.5*max(x)))+1e-15) for x in self.datset['noisy_phens']]
  #self.phens = torch.sigmoid(torch.tensor(self.datset['noisy_phens']))
  self.phens = self.datset['noisy_phens']
  self.genotypes = self.datset['genotypes']
  self.weights = self.datset['weights']
  self.data_file = data_file
  self.n_phens=n_phens

 def __len__(self):
  return len(self.phens)

 def __getitem__(self,idx):
  phenotypes=torch.tensor(self.phens[idx][:self.n_phens],dtype=torch.float32)
  genotype=torch.tensor(self.genotypes[idx],dtype=torch.float32)
  return phenotypes, genotype

#custom layer that passes the max of an array and EPS elsewhere
class max_layer(nn.Module):
 def __init__(self):
  super().__init__()

 def forward(self,X):
  c=torch.ones(size=X.shape)*1e-15
  indices=torch.max(X,2)[1]
  for n in range(len(indices)): c[n,indices[n]] = X[n,indices[n]]
  return c

# error definition
def mean_absolute_percentage_error(y_true, y_pred):
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets

#dataset_path = '/home/dmets/git/arcadia-genotype-phenotype-map-nn/data/n_3000_nlp_30/'
#dataset_path = '/home/dmets/git/dm_archive/nn_test/data/gen_sim_dat/'
#dataset_path = '/home/dmets/git/arcadia-genotype-phenotype-map-nn/data/p_i_sweep_10_loci_30_phens/test_5/'
dataset_path = '/home/dmets/git/arcadia-genotype-phenotype-map-nn/data/n_3000_nlip_10/'

train_data = phen_dataset(dataset_path+'train.pk',n_phens=vabs.n_phens_to_analyze)
test_data = phen_dataset(dataset_path+'test.pk',n_phens=vabs.n_phens_to_analyze)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#load_data
# how many samples per batch to load
batch_size = vabs.batch_size
num_workers = vabs.n_cpu

# prepare data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
data_iter=iter(train_loader)



n_env=vabs.n_env
n_locs=vabs.n_locs
n_alleles=vabs.n_alleles
n_allelic_states=n_alleles*n_locs
latent_dim=n_allelic_states+n_env


#encoder
class Q_net(nn.Module):
 def __init__(self,phen_dim=None,N=None):
  super().__init__()
  if N is None: N=vabs.e_hidden_dim
  if phen_dim is None: phen_dim=vabs.n_phens_to_analyze

  batchnorm_momentum=vabs.batchnorm_momentum

  self.encoder = nn.Sequential(
   nn.Linear(in_features=phen_dim, out_features=N),
   nn.BatchNorm1d(N,momentum=batchnorm_momentum),
   nn.LeakyReLU(0.01,inplace=True),
   nn.Linear(in_features=N, out_features=N),
   #nn.BatchNorm1d(N,momentum=batchnorm_momentum),
   nn.LeakyReLU(0.01,inplace=True)
  )

  #output layers
  self.cat_layer= nn.Sequential(nn.Linear(N,n_allelic_states))
  #self.cat_layer2=nn.Sequential(nn.Softmax(dim=2),max_layer(),nn.Flatten())
  self.cat_layer2=nn.Sequential(nn.Softmax(dim=2),nn.Flatten())
  #self.cat_layer3=nn.Sequential(nn.Sigmoid())
  self.norm_layer= nn.Sequential(nn.Linear(N, n_env))

 def forward(self,x):
  x=self.encoder(x)
  cat_out=self.cat_layer(x)
  cat_out=torch.reshape(cat_out,(batch_size,n_locs,n_alleles))
  cat_out=self.cat_layer2(cat_out)
  #cat_out=self.cat_layer3(cat_out)
  norm_out=self.norm_layer(x)
  return(cat_out,norm_out)

#decoder
class P_net(nn.Module):
 def __init__(self,phen_dim=None,N=None):
  if N is None: N=vabs.d_hidden_dim
  if phen_dim is None: phen_dim=vabs.n_phens

  n_env=vabs.n_env
  n_allelic_states=vabs.n_locs*vabs.n_alleles
  latent_dim=n_allelic_states+n_env

  batchnorm_momentum=vabs.batchnorm_momentum

  super().__init__()
  self.decoder = nn.Sequential(
   nn.Linear(in_features=latent_dim, out_features=N),
   nn.BatchNorm1d(N,momentum=batchnorm_momentum),
   nn.LeakyReLU(0.2),
   #nn.Linear(in_features=N, out_features=N),
   #nn.BatchNorm1d(N,momentum=batchnorm_momentum),
   nn.Linear(in_features=N, out_features=phen_dim),
   nn.LeakyReLU(0.01)
  )

 def forward(self,cat_in,norm_in):
  input=torch.cat((cat_in,norm_in),-1)
  x=self.decoder(input)
  return(x)

#cat discriminator
class D_net_cat(nn.Module):
 def __init__(self):
  super().__init__()

  n_allelic_states=vabs.n_locs*vabs.n_alleles
  n_hidden=vabs.hidden_dim_dg

  self.discriminator = nn.Sequential(
  nn.Linear(in_features=n_allelic_states, out_features=n_hidden),
  nn.BatchNorm1d(n_hidden),
  nn.LeakyReLU(0.2),
  nn.Linear(in_features=n_hidden, out_features=1),
  nn.Sigmoid()
 )

 def forward(self,x):
  x=self.discriminator(x)
  return(x)

#gauss discriminator
class D_net_gauss(nn.Module):
 def __init__(self):
  super().__init__()

  n_env=vabs.n_env
  n_hidden=vabs.hidden_dim_de

  self.discriminator = nn.Sequential(
  nn.Linear(in_features=n_env, out_features=n_hidden),
  nn.BatchNorm1d(n_hidden),
  nn.LeakyReLU(0.2),
  nn.Linear(in_features=n_hidden, out_features=1),
  nn.Sigmoid())

 def forward(self,x):
  x=self.discriminator(x)
  return(x)



#main
EPS = 1e-15
Q=Q_net()
P=P_net()
D=D_net_cat()
DG=D_net_gauss()

#put everything on the GPU if it is there
Q.to(device)
P.to(device)
D.to(device)
DG.to(device)

# Set learning rates
gen_lr = vabs.lr_g
reg_lr = vabs.lr_r
reg_gauss_lr = vabs.lr_r_g
reg_cat_lr = vabs.lr_r_c

#adam betas
adam_b=(vabs.b1,vabs.b2)

#encode/decode optimizers
optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr, betas=adam_b)
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr, betas=adam_b)

#regularizing optimizers
optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr, betas=adam_b)
optim_D_cat = torch.optim.Adam(D.parameters(), lr=reg_cat_lr)
optim_D_gauss = torch.optim.Adam(DG.parameters(), lr=reg_gauss_lr)


num_epochs=vabs.n_epochs

torch.manual_seed(47)

#Train
n_phens = vabs.n_phens_to_analyze
n_phens_pred = vabs.n_phens_to_predict

rcon_loss=[]
d_cat_loss=[]
d_gauss_loss=[]
g_loss=[]

start_time=tm.time()

for n in range(num_epochs):
 for i,(phens,_gen) in enumerate(train_loader):

  phens = phens.to(device) #move data to GPU if it is there

  batch_size = phens.shape[0] #redefine batch size here to allow for incomplete batches

  #reconstruction loss
  P.zero_grad()
  Q.zero_grad()
  D.zero_grad()
  DG.zero_grad()

  noise_phens = phens + (vabs.sd_noise**0.5)*torch.randn(phens.shape).to(device)

  z_cat_sample, z_gauss_sample = Q(noise_phens)   #encode to z
  X_sample = P(z_cat_sample, z_gauss_sample) #decode to X reconstruction
  #recon_loss = F.binary_cross_entropy(X_sample+EPS,phens+EPS)
  recon_loss = F.mse_loss(X_sample+EPS,phens+EPS)
  rcon_loss.append(float(recon_loss.detach()))
  recon_loss.backward()
  optim_P.step()
  optim_Q_enc.step()

  # Discriminator cat
  Q.eval()
  real_cat_dist = (1.0/n_alleles)*np.ones((n_locs,n_alleles))
  real_cat_dist = torch.distributions.OneHotCategorical(torch.tensor(real_cat_dist))
  z_real_cat = Variable(torch.flatten(real_cat_dist.sample(sample_shape=torch.Size([batch_size])),start_dim=1))
  z_real_cat = z_real_cat.to(device)
  z_real_cat = z_real_cat.float()
  D_real_cat = D(z_real_cat)
  z_fake_cat, _ = Q(phens)
  D_fake_cat = D(z_fake_cat)
  D_cat_loss = -torch.mean(torch.log(D_real_cat + EPS) + torch.log(D_fake_cat + EPS))
  d_cat_loss.append(float(D_cat_loss.detach()))
  D_cat_loss.backward()
  optim_D_cat.step()

  #Discriminator gauss
  Q.eval()
  _, z_fake_gauss = Q(phens)
  z_real_gauss = Variable(torch.randn(phens.size()[0], n_env) * 5.)
  z_real_gauss = z_real_gauss.to(device)
  D_real_gauss = DG(z_real_gauss)
  D_fake_gauss = DG(z_fake_gauss)
  D_gauss_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(D_fake_gauss + EPS))
  d_gauss_loss.append(float(D_gauss_loss.detach()))
  D_gauss_loss.backward()
  optim_D_gauss.step()

  #generator()
  Q.train()

  lambda_cat=1
  lambda_gauss=0.1

  z_fake_cat, z_fake_gauss = Q(phens)
  D_fake_cat = D(z_fake_cat)
  D_fake_gauss = DG(z_fake_gauss)
  G_loss = -torch.mean(lambda_cat*(torch.log(D_fake_cat + EPS)) + lambda_gauss*(torch.log(D_fake_gauss + EPS)))
  g_loss.append(float(G_loss.detach()))
  G_loss.backward()
  optim_Q_gen.step()

 cur_time = (tm.time()-start_time)
 start_time=tm.time()
 print('Epoch num: '+str(n)+' batchno '+str(i)+' r_con_loss: '+str(rcon_loss[-1])+' g_loss: '+str(g_loss[-1])+' d_cat_loss: '+str(d_cat_loss[-1])+' d_gauss_loss: '+str(d_gauss_loss[-1])+' epoch duration: '+str(cur_time))

train_labs=[]

#plot loss funciton output
plt.plot(rcon_loss)
plt.plot(d_cat_loss)
plt.plot(g_loss)
plt.plot(d_gauss_loss)
plt.legend(['rcon loss','d_cat loss','g loss','d_gauss loss'])
plt.savefig(dataset_path+'loss_plot.svg')
plt.close()

#save the Encoder
torch.save(Q.state_dict(), dataset_path+'Q_encoder_weights_denoise.pt')
#save the Decoder
torch.save(P.state_dict(), dataset_path+'P_encoder_weights_denoise.pt')


#examine test data and plot test set statistics
phen_encodings=[]
phens=[]
z_samples_cat=[]
z_samples_gauss=[]
gens=[]
for dat in test_loader:
 ph,gt=dat
 gens+=list(gt.detach().cpu().numpy())
 ph=ph.to(device)
 batch_size=ph.shape[0]
 z_sample_cat,z_sample_gauss = Q(ph)
 X_sample = P(z_sample_cat,z_sample_gauss)
 phens+=list(ph.detach().cpu().numpy())
 phen_encodings+=list(X_sample.detach().cpu().numpy())
 z_samples_cat+=list(z_sample_cat.detach().cpu().numpy())
 z_samples_gauss+=list(z_sample_gauss.detach().cpu().numpy())


#save genotype and latent representation information:
z_samples_cat = np.array(z_samples_cat)
latent_genes = z_samples_cat.T
gens=np.array(gens)
gens = gens.reshape(len(gens),len(gens[0])*len(gens[0][0])).T
print(latent_genes[0][:30])
print(gens[0][:30])
pk.dump([gens,latent_genes],open(dataset_path+'gens_latent_denoise.pk','bw'))

latent_sum = [sum(x) for x in latent_genes]
plt.hist(latent_sum,bins=100)
plt.savefig(dataset_path+'latent_sum_historgram.svg')
plt.close()

filtered_latent_genes = [x for x in latent_genes if sum(x)>0 and sum(x)<600]

prs = []
for x in filtered_latent_genes:
 for y in filtered_latent_genes:
  prs.append(sc.stats.pearsonr(x,y)[0])
prs = np.array(prs).reshape([len(filtered_latent_genes),len(filtered_latent_genes)])

plt.imshow(prs)
plt.savefig(dataset_path+'pearsons_r_matrix.svg')
plt.close()

phens=np.array(phens).T
phen_encodings=np.array(phen_encodings).T

print(phens[0][:30])
print(phen_encodings[0][:30])
print(z_samples_cat[0][:30])
print(z_samples_cat[1][:30])
print(z_samples_cat[2][:30])
print(z_samples_gauss[0][:30])



fig = plt.figure
ax = plt.axes(projection='3d')
sample = z_samples_cat[0]
dat = np.array(sample).reshape(int(len(sample)/n_alleles),n_alleles)
for n in dat: ax.scatter3D(n[0],n[1],n[2])
ax.view_init(-160,60)
plt.savefig(dataset_path+'3d_allelic_state_latent_denoise.svg')
plt.close()

plt.hist(z_samples_cat[0], bins = 100)
plt.savefig(dataset_path+'2d_allelic_state_latent_denoise.svg')
plt.close()

for n in range(len(phens)):
 plt.plot(phens[n],phen_encodings[n],'o')
plt.xlabel('real')
plt.ylabel('predicted')
plt.savefig(dataset_path+'phen_real_pred_denoise.svg')
plt.close()

cors=[sc.stats.pearsonr(phens[n],phen_encodings[n])[0] for n in range(len(phens))]
plt.hist(cors,bins=20)
plt.savefig(dataset_path+'phen_real_pred_pearsonsr_denoise.svg')
plt.close()
print(cors)

errs=[mean_squared_error(phens[n],phen_encodings[n]) for n in range(len(phens[:n_phens_pred]))]
print(errs)
plt.hist(errs,bins=20)
plt.savefig(dataset_path+'phen_real_pred_mse_denoise.svg')
plt.close()
#plt.show()

errs=[mean_absolute_percentage_error(phens[n],phen_encodings[n]) for n in range(len(phens[:n_phens_pred]))]
print(errs)
plt.hist(errs,bins=20)
plt.savefig(dataset_path+'phen_real_pred_mpe_denoise.svg')
plt.close()
#plt.show()


'''z_samples_cat = np.array(z_samples_cat)
latent_genes = z_samples_cat.T

z_samples_cat = z_samples_cat.reshape(500,-1,3)
latent_genes = z_samples_cat
latent_genes = z_samples_cat.transpose((1,0,2))
#latent_genes = latent_genes.reshape(vabs.n_locs,-1).T
print(latent_genes[0][:30])
latent_genes=[[list(x).index(max(x)) for x in y] for y in latent_genes]
print(latent_genes[0][:30])

gens = np.array(gens)
gens = gens.transpose((1,0,2))
#gens = gens.reshape(vabs.n_locs,-1).T
print(gens[0][:30])
gens = [[list(x).index(max(x)) for x in y] for y in gens]'''

'''gens=np.array(gens)
gens = gens.reshape(len(gens),len(gens[0])*len(gens[0][0])).T


print(latent_genes[0][:30])
print(gens[0][:30])

pk.dump([gens,latent_genes],open(dataset_path+'gens_latent_testt.pk','bw'))'''


'''mis = []
for x in latent_genes[:10]:
 sub_mis=[]
 for y in gens: sub_mis.append(mutual_info_score(x,y))
 mis.append(sub_mis)

for mi in mis:
 plt.plot(mi)
plt.show()

plt.hist([np.max(x) for x in mis],bins=25)
plt.show()'''

