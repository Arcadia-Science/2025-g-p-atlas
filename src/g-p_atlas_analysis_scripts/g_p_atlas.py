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
    help="flag whether to detect allele-allele interactions. Expects 'no' or 'yes.' ",
)
args.add_argument(
    "--interaction_threshold",
    type=float,
    default=0.05,
    help="threshold for considering an allele-allele interaction significant",
)
args.add_argument(
    "--max_loci_interactions",
    type=int,
    default=100,
    help="maximum number of loci to test for allele interactions (to limit computation)",
)
args.add_argument(
    "--loci_interaction_indices",
    type=str,
    default="",
    help="comma-separated list of loci indices or ranges (e.g., '1,2,3' or '10-20,30-40' or '1,5-10') to include in interaction analysis",
)
args.add_argument(
    "--max_phenotypes_for_interactions",
    type=int,
    default=3,
    help="maximum number of phenotypes to analyze for interactions",
)
args.add_argument(
    "--phenotype_interaction_indices",
    type=str,
    default="",
    help="comma-separated list of phenotype indices to analyze for interactions (e.g., '0,2,5'). If not provided, the first 'max_phenotypes_for_interactions' phenotypes will be used.",
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


# Helper functions for gene-gene interactions
def detect_pairwise_interactions(model, P, genotypes, phenotypes, phenotype_idx=0, threshold=0.05):
    """
    Detects pairwise interactions between allelic states by measuring the non-additive 
    effects on phenotype predictions.
    
    This method works by calculating the difference between the expected additive effect
    of modifying two alleles independently and the actual effect of modifying them together.
    A non-zero difference indicates an allele-allele interaction (epistasis).
    
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
        List of (allele1_info, allele2_info, interaction_data) tuples where:
        - allele_info is (locus, allele_idx_within_locus, absolute_idx)
        - interaction_data contains normalized and raw interaction values plus main effects
    """
    # Determine the total number of allelic states
    total_alleles = genotypes.shape[1]
    n_loci = total_alleles // vabs.n_alleles
    
    # Move models to evaluation mode
    model.eval()
    P.eval()
    
    # Store the device
    device = next(model.parameters()).device
    
    # Initialize interactions list
    interactions = []
    
    # Parse the loci indices string if provided
    loci_to_test = []
    if vabs.loci_interaction_indices:
        # Parse the comma-separated list of indices and ranges
        for part in vabs.loci_interaction_indices.split(','):
            if '-' in part:
                # Handle range (e.g., "10-20")
                start, end = map(int, part.split('-'))
                loci_to_test.extend(range(start, end + 1))
            else:
                # Handle single index
                loci_to_test.append(int(part))
        
        # Remove duplicates and sort
        loci_to_test = sorted(set(loci_to_test))
        
        # Filter out invalid indices
        loci_to_test = [i for i in loci_to_test if 0 <= i < n_loci]
        
        print(f"Testing interactions for {len(loci_to_test)} specified loci indices out of {n_loci} total")
    else:
        # Limit the number of loci to test for interactions if no specific indices provided
        max_loci = min(n_loci, vabs.max_loci_interactions)
        loci_to_test = list(range(max_loci))
        if max_loci < n_loci:
            print(f"Testing interactions for first {max_loci} loci out of {n_loci} total")
    
    # For each pair of alleles (across different loci)
    total_pairs = (len(loci_to_test) * (len(loci_to_test) - 1)) // 2
    pair_count = 0
    
    for idx1, i in enumerate(loci_to_test):
        i_start = i * vabs.n_alleles
        i_end = i_start + vabs.n_alleles
        
        for idx2, j in enumerate(loci_to_test[idx1 + 1:], idx1 + 1):
            j_start = j * vabs.n_alleles
            j_end = j_start + vabs.n_alleles
            
            # Print progress periodically
            pair_count += 1
            if pair_count % 1000 == 0 or pair_count == total_pairs:
                print(f"Testing locus pair {pair_count}/{total_pairs} ({(pair_count/total_pairs)*100:.1f}%)")
            
            # For each specific allele at locus i
            for i_allele in range(i_start, i_end):
                # For each specific allele at locus j
                for j_allele in range(j_start, j_end):
                    # Create modified genotypes for measuring individual effects
                    genotypes_i_modified = genotypes.clone()
                    genotypes_j_modified = genotypes.clone()
                    genotypes_both_modified = genotypes.clone()
                    
                    # Set specific allele to 0 (absent) to test its effect
                    # We're turning off a specific allele rather than the whole locus
                    genotypes_i_modified[:, i_allele] = 0
                    genotypes_j_modified[:, j_allele] = 0
                    genotypes_both_modified[:, i_allele] = 0
                    genotypes_both_modified[:, j_allele] = 0
                    
                    # Get predictions
                    with torch.no_grad():
                        pred_baseline = P(model(genotypes.to(device)))[:, phenotype_idx]
                        pred_i_modified = P(model(genotypes_i_modified.to(device)))[:, phenotype_idx]
                        pred_j_modified = P(model(genotypes_j_modified.to(device)))[:, phenotype_idx]
                        pred_both_modified = P(model(genotypes_both_modified.to(device)))[:, phenotype_idx]
                    
                    # Calculate effects of removing each allele (main effects)
                    effect_i = pred_baseline - pred_i_modified
                    effect_j = pred_baseline - pred_j_modified
                    
                    # Store mean effects as scalars for easier analysis
                    effect_i_mean = effect_i.mean().item()
                    effect_j_mean = effect_j.mean().item()
                    
                    # Expected additive effect
                    expected_both_effect = effect_i + effect_j
                    
                    # Actual effect of removing both alleles
                    actual_both_effect = pred_baseline - pred_both_modified
                    
                    # Interaction strength - the difference between actual and expected effects
                    interaction_raw = actual_both_effect - expected_both_effect
                    
                    # Calculate mean of raw interaction strength
                    interaction_raw_mean = interaction_raw.mean().item()
                    
                    # Calculate absolute value of mean interaction (for significance testing)
                    interaction_abs_mean = torch.abs(interaction_raw).mean().item()
                    
                    # Normalize by the standard deviation of phenotype values for better comparability
                    pheno_std = phenotypes[:, phenotype_idx].std().item()
                    normalized_strength = interaction_abs_mean / (pheno_std + EPS)
                    
                    # If interaction is significant, add to list
                    if normalized_strength > threshold:
                        # Prepare interaction data object
                        interaction_data = {
                            # Normalized strength (for significance testing and sorting)
                            'normalized_strength': normalized_strength,
                            
                            # Raw interaction values
                            'raw_interaction_mean': interaction_raw_mean,
                            'raw_interaction_abs_mean': interaction_abs_mean,
                            
                            # Main effects for each allele
                            'allele1_main_effect': effect_i_mean,
                            'allele2_main_effect': effect_j_mean,
                            
                            # Phenotype standard deviation (for reference)
                            'phenotype_std': pheno_std
                        }
                        
                        # Store allele indices and their loci for better interpretability
                        interactions.append((
                            (i, i_allele - i_start, i_allele),  # (locus, allele_idx_within_locus, absolute_idx)
                            (j, j_allele - j_start, j_allele),  # (locus, allele_idx_within_locus, absolute_idx)
                            interaction_data
                        ))
    
    # Sort by normalized interaction strength
    interactions.sort(key=lambda x: x[2]['normalized_strength'], reverse=True)
    
    return interactions


def plot_interaction_network(interaction_importance, locus_names=None, phenotype_name=None,
                         threshold=0.05, max_nodes=50):
    """
    Visualize allele-allele interactions as a network.
    
    Parameters:
    -----------
    interaction_importance : list of tuples
        List of (allele1_info, allele2_info, interaction_data) tuples
        where:
        - allele_info is (locus, allele_idx_within_locus, absolute_idx)
        - interaction_data is a dictionary with interaction metrics
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
    filtered_interactions = [(a1, a2, data) for a1, a2, data in interaction_importance 
                             if data['normalized_strength'] > threshold]
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes and edges
    all_alleles = set()
    for allele1, allele2, interaction_data in filtered_interactions:
        # Use the absolute allele index as the node ID
        a1_id = allele1[2]  # absolute_idx
        a2_id = allele2[2]  # absolute_idx
        all_alleles.add(a1_id)
        all_alleles.add(a2_id)
    
    # Limit number of nodes if necessary
    if len(all_alleles) > max_nodes:
        # Find the max_nodes most important alleles
        allele_importance = {}
        for allele1, allele2, interaction_data in filtered_interactions:
            a1_id = allele1[2]
            a2_id = allele2[2]
            importance = interaction_data['normalized_strength']
            allele_importance[a1_id] = allele_importance.get(a1_id, 0) + importance
            allele_importance[a2_id] = allele_importance.get(a2_id, 0) + importance
        
        top_alleles = sorted(allele_importance.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_alleles = set(x[0] for x in top_alleles)
        filtered_interactions = [(a1, a2, data) for a1, a2, data in filtered_interactions 
                            if a1[2] in top_alleles and a2[2] in top_alleles]
    
    # Add nodes and edges
    for allele1, allele2, interaction_data in filtered_interactions:
        a1_id = allele1[2]  # absolute_idx
        a2_id = allele2[2]  # absolute_idx
        
        # Create more descriptive node names
        if locus_names is not None:
            a1_name = f"{locus_names[allele1[0]]}_Allele{allele1[1]}"
            a2_name = f"{locus_names[allele2[0]]}_Allele{allele2[1]}"
        else:
            a1_name = f"Locus{allele1[0]}_Allele{allele1[1]}"
            a2_name = f"Locus{allele2[0]}_Allele{allele2[1]}"
        
        # Add node if not already present
        if not G.has_node(a1_id):
            G.add_node(a1_id, name=a1_name, locus=allele1[0], allele=allele1[1],
                       main_effect=interaction_data['allele1_main_effect'])
        
        if not G.has_node(a2_id):
            G.add_node(a2_id, name=a2_name, locus=allele2[0], allele=allele2[1],
                       main_effect=interaction_data['allele2_main_effect'])
        
        # Add edge with full interaction data
        G.add_edge(a1_id, a2_id, 
                   weight=interaction_data['normalized_strength'],
                   raw_interaction=interaction_data['raw_interaction_mean'])
    
    # Set up plot
    plt.figure(figsize=(14, 14))
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get edge weights for colors
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    if not edge_weights:
        plt.title(f"No significant allele-allele interactions found for {phenotype_name}")
        return plt.gcf()
    
    # Normalize weights for colormap
    norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
    
    # Size nodes by their main effect (absolute value)
    node_sizes = [300 + 1000 * abs(G.nodes[n]['main_effect']) for n in G.nodes()]
    
    # Color nodes by locus
    node_colors = [G.nodes[n]['locus'] % 20 for n in G.nodes()]  # Cycle through 20 colors
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                           node_size=node_sizes, 
                           alpha=0.8, 
                           node_color=node_colors, 
                           cmap=plt.cm.tab20)
    
    # Draw edges with color based on weight
    edges = nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_weights, 
                                  edge_cmap=plt.cm.viridis, edge_vmin=min(edge_weights), 
                                  edge_vmax=max(edge_weights))
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['name'] for n in G.nodes()}, 
                        font_size=9, font_family='sans-serif')
    
    # Add colorbar
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Normalized Interaction Strength')
    
    # Set title
    if phenotype_name is not None:
        plt.title(f'Allele-Allele Interaction Network for {phenotype_name}', fontsize=16)
    else:
        plt.title('Allele-Allele Interaction Network', fontsize=16)
    
    # Add legend for node size
    plt.figtext(0.02, 0.02, "Node size = Main effect magnitude", fontsize=10)
    
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()


def analyze_gene_interaction_network(model, P, test_loader, dataset_path):
    """
    Perform final analysis of allele-allele interactions and save results.
    
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
    print("Loading test data for allele interaction analysis...")
    
    # Use a more efficient approach to collect test data
    # This reduces memory usage and potential file handle issues
    device = next(model.parameters()).device
    
    # Collect all test data more efficiently
    # Process in smaller batches to avoid memory issues
    all_genotypes = []
    all_phenotypes = []
    
    # Create a loader with no multiprocessing for this specific step
    # This avoids file descriptor issues
    single_worker_loader = torch.utils.data.DataLoader(
        dataset=test_loader.dataset, 
        batch_size=16,  # Use a manageable batch size
        num_workers=0,  # No multiprocessing to avoid file handle issues
        shuffle=False
    )
    
    with torch.no_grad():  # Save memory by not tracking gradients
        for dat in single_worker_loader:
            ph, gt = dat
            gt = gt[:, : vabs.n_loci_measured * vabs.n_alleles]
            # Move to CPU to avoid GPU memory issues
            all_genotypes.append(gt.cpu())
            all_phenotypes.append(ph.cpu())
    
    # Concatenate once at the end to save memory
    all_genotypes = torch.cat(all_genotypes, dim=0)
    all_phenotypes = torch.cat(all_phenotypes, dim=0)
    
    print(f"Processed {len(all_genotypes)} test samples for allele interaction analysis")
    
    # Determine which phenotypes to analyze
    phenotypes_to_analyze = []
    if vabs.phenotype_interaction_indices:
        # Parse the comma-separated list of indices
        try:
            phenotype_indices = [int(idx.strip()) for idx in vabs.phenotype_interaction_indices.split(',')]
            # Filter out invalid indices
            total_phenotypes = all_phenotypes.shape[1]
            phenotypes_to_analyze = [idx for idx in phenotype_indices if 0 <= idx < total_phenotypes]
            
            if not phenotypes_to_analyze:
                print(f"Warning: No valid phenotype indices provided. Using default.")
                phenotypes_to_analyze = list(range(min(vabs.max_phenotypes_for_interactions, total_phenotypes)))
            else:
                print(f"Analyzing interactions for {len(phenotypes_to_analyze)} specified phenotypes: {phenotypes_to_analyze}")
        except ValueError:
            print(f"Error parsing phenotype indices. Using default.")
            phenotypes_to_analyze = list(range(min(vabs.max_phenotypes_for_interactions, all_phenotypes.shape[1])))
    else:
        # Use default sequential phenotypes up to the maximum
        phenotypes_to_analyze = list(range(min(vabs.max_phenotypes_for_interactions, all_phenotypes.shape[1])))
        print(f"Analyzing interactions for the first {len(phenotypes_to_analyze)} phenotypes")
    
    all_interaction_data = {}
    
    # Also collect main effects of alleles across all phenotypes
    allele_main_effects = {}
    
    for i, phen_idx in enumerate(phenotypes_to_analyze):
        print(f"Analyzing allele interactions for phenotype {phen_idx} ({i+1}/{len(phenotypes_to_analyze)})")
        
        # Calculate interactions for this phenotype
        interactions = detect_pairwise_interactions(
            model, P, all_genotypes, all_phenotypes, 
            phenotype_idx=phen_idx, threshold=vabs.interaction_threshold)
        
        # Convert to dictionary format for compatibility
        # Create a unique key for each allele pair
        interaction_dict = {}
        
        # Track allele main effects for this phenotype
        this_phenotype_main_effects = {}
        
        for a1, a2, interaction_data in interactions:
            # Use the absolute indices as the key
            key = (a1[2], a2[2])
            
            # Store the full information including detailed allele info and interaction data
            interaction_dict[key] = {
                'allele1': a1,
                'allele2': a2,
                'interaction_data': interaction_data
            }
            
            # Track main effects for each allele
            a1_id = a1[2]  # absolute index
            a2_id = a2[2]
            
            # Update main effects for this phenotype
            if a1_id not in this_phenotype_main_effects:
                this_phenotype_main_effects[a1_id] = {
                    'allele_info': a1,
                    'main_effect': interaction_data['allele1_main_effect']
                }
            
            if a2_id not in this_phenotype_main_effects:
                this_phenotype_main_effects[a2_id] = {
                    'allele_info': a2,
                    'main_effect': interaction_data['allele2_main_effect']
                }
        
        # Add main effects to global tracking
        allele_main_effects[phen_idx] = this_phenotype_main_effects
        
        # Plot network
        plt.figure(figsize=(14, 12))
        fig = plot_interaction_network(
            interactions, phenotype_name=f"Phenotype {phen_idx}", threshold=vabs.interaction_threshold)
        
        # Save plot
        plt.savefig(dataset_path + f"allele_interaction_network_phenotype_{phen_idx}.svg")
        plt.savefig(dataset_path + f"allele_interaction_network_phenotype_{phen_idx}.png")
        plt.close()
        
        # Store data
        all_interaction_data[phen_idx] = interaction_dict
        
        # Report the number of significant interactions found
        print(f"Found {len(interactions)} significant interactions for phenotype {phen_idx} (threshold: {vabs.interaction_threshold})")
    
    # Save all interaction data
    with open(dataset_path + "allele_allele_interactions.pk", "wb") as f:
        output_data = {
            'interactions': all_interaction_data,
            'main_effects': allele_main_effects
        }
        pk.dump(output_data, f)
    
    # Create a summary of the most significant interactions across phenotypes
    interaction_summary = {}
    
    for phen_idx, interactions in all_interaction_data.items():
        for key, data in interactions.items():
            if key not in interaction_summary:
                interaction_summary[key] = []
            
            # Include normalized strength for summary
            interaction_summary[key].append(
                (phen_idx, data['interaction_data']['normalized_strength'])
            )
    
    # Find interactions affecting multiple phenotypes
    multi_phenotype_interactions = {k: v for k, v in interaction_summary.items() if len(v) > 1}
    
    # Sort by total importance across phenotypes
    sorted_multi = sorted(multi_phenotype_interactions.items(), 
                        key=lambda x: sum(imp for _, imp in x[1]), 
                        reverse=True)
    
    # Save top multi-phenotype interactions
    with open(dataset_path + "allele_multi_phenotype_interactions.pk", "wb") as f:
        pk.dump(sorted_multi, f)
    
    # Create a human-readable report of the most significant allele interactions
    with open(dataset_path + "allele_interaction_report.txt", "w") as f:
        f.write("Allele-Allele Interaction Analysis Report\n")
        f.write("=======================================\n\n")
        
        # Report per phenotype
        for phen_idx, interactions in all_interaction_data.items():
            f.write(f"Phenotype {phen_idx}:\n")
            f.write("--------------\n")
            
            # Sort interactions by normalized strength
            sorted_interactions = sorted(interactions.items(), 
                                         key=lambda x: x[1]['interaction_data']['normalized_strength'], 
                                         reverse=True)
            
            # Report top interactions (limit to 20 for readability)
            for i, (key, data) in enumerate(sorted_interactions[:20]):
                allele1 = data['allele1']
                allele2 = data['allele2']
                interaction_data = data['interaction_data']
                
                # Create detailed interaction report
                f.write(f"{i+1}. Locus{allele1[0]}_Allele{allele1[1]} × Locus{allele2[0]}_Allele{allele2[1]}:\n")
                f.write(f"   Normalized strength: {interaction_data['normalized_strength']:.4f}\n")
                f.write(f"   Raw interaction: {interaction_data['raw_interaction_mean']:.4f}\n")
                f.write(f"   Main effects: {interaction_data['allele1_main_effect']:.4f}, {interaction_data['allele2_main_effect']:.4f}\n")
                f.write("\n")
            
            f.write("\n")
        
        # Report multi-phenotype interactions
        f.write("Multi-Phenotype Interactions:\n")
        f.write("---------------------------\n")
        
        for i, (key, phenotypes) in enumerate(sorted_multi[:20]):
            a1_idx, a2_idx = key
            
            # Find any interaction data to get the allele info
            for phen_idx, interactions in all_interaction_data.items():
                if key in interactions:
                    data = interactions[key]
                    allele1 = data['allele1']
                    allele2 = data['allele2']
                    interaction_data = data['interaction_data']
                    
                    f.write(f"{i+1}. Locus{allele1[0]}_Allele{allele1[1]} × Locus{allele2[0]}_Allele{allele2[1]}:\n")
                    f.write(f"   Total normalized importance: {sum(imp for _, imp in phenotypes):.4f}\n")
                    f.write(f"   Raw interaction (phenotype {phen_idx}): {interaction_data['raw_interaction_mean']:.4f}\n")
                    f.write(f"   Main effects (phenotype {phen_idx}): {interaction_data['allele1_main_effect']:.4f}, {interaction_data['allele2_main_effect']:.4f}\n")
                    f.write(f"   Affects phenotypes: {', '.join(f'{p}({imp:.4f})' for p, imp in phenotypes)}\n")
                    f.write("\n")
                    break
        
        # Create a section for main effects
        f.write("\nAllele Main Effects:\n")
        f.write("==================\n\n")
        
        for phen_idx, effects in allele_main_effects.items():
            f.write(f"Phenotype {phen_idx}:\n")
            f.write("--------------\n")
            
            # Get top alleles by absolute effect
            sorted_effects = sorted(effects.items(), 
                                    key=lambda x: abs(x[1]['main_effect']), 
                                    reverse=True)
            
            # Report top alleles (limit to 20 for readability)
            for i, (allele_id, effect_data) in enumerate(sorted_effects[:20]):
                allele_info = effect_data['allele_info']
                main_effect = effect_data['main_effect']
                
                f.write(f"{i+1}. Locus{allele_info[0]}_Allele{allele_info[1]}: {main_effect:.4f}\n")
            
            f.write("\n")
    
    # Create a csv file for easier data analysis in spreadsheets/R/etc.
    with open(dataset_path + "allele_interactions.csv", "w") as f:
        # Write header
        f.write("phenotype,locus1,allele1_idx,allele1_abs_idx,locus2,allele2_idx,allele2_abs_idx,")
        f.write("normalized_strength,raw_interaction_mean,raw_interaction_abs_mean,")
        f.write("allele1_main_effect,allele2_main_effect,phenotype_std\n")
        
        # Write data for each interaction
        for phen_idx, interactions in all_interaction_data.items():
            for key, data in interactions.items():
                allele1 = data['allele1']
                allele2 = data['allele2']
                idata = data['interaction_data']
                
                # Write one row per interaction
                f.write(f"{phen_idx},")
                f.write(f"{allele1[0]},{allele1[1]},{allele1[2]},")
                f.write(f"{allele2[0]},{allele2[1]},{allele2[2]},")
                f.write(f"{idata['normalized_strength']},")
                f.write(f"{idata['raw_interaction_mean']},")
                f.write(f"{idata['raw_interaction_abs_mean']},")
                f.write(f"{idata['allele1_main_effect']},")
                f.write(f"{idata['allele2_main_effect']},")
                f.write(f"{idata['phenotype_std']}\n")
    
    print(f"Saved allele interaction analysis results to {dataset_path}")
    
    # Clean up to release memory
    del all_genotypes
    del all_phenotypes
    
    return all_interaction_data


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
# Reduce number of workers to avoid too many open files
# For training we can use more workers as it's more CPU intensive
train_num_workers = min(vabs.n_cpu, 8)  # Limit to 8 workers
# For testing, use fewer workers to avoid file descriptor issues
test_num_workers = 2

# prepare data loaders
train_loader_pheno = torch.utils.data.DataLoader(
    dataset=train_data_pheno, batch_size=batch_size, num_workers=train_num_workers, shuffle=True
)
test_loader_pheno = torch.utils.data.DataLoader(
    dataset=test_data_pheno, batch_size=1, num_workers=test_num_workers, shuffle=True
)

train_loader_geno = torch.utils.data.DataLoader(
    dataset=train_data_geno, batch_size=batch_size, num_workers=train_num_workers, shuffle=True
)
test_loader_geno = torch.utils.data.DataLoader(
    dataset=test_data_geno, batch_size=1, num_workers=test_num_workers, shuffle=True
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

# Detection of allele-allele interactions
if vabs.detect_interactions == "yes":
    print("Detecting allele-allele interactions...")
    
    # Use the comprehensive analysis function
    analyze_gene_interaction_network(GQ, P, test_loader_geno, dataset_path)

# Save and close stats
pk.dump(stats_aggregator, out_stats)
out_stats.close()