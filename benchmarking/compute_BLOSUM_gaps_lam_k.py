import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import argparse
import sys
import os
import csv
import pickle
from utils import *
from gap_search_via_aln_benchmark import *

sys.path.append(os.path.abspath('../blosum_construction/'))
from build_BLOSUM import *
from compute_lambda_and_k import *


# should use on GPU!
# input is one-hot dictionary of all sequence in alphabet: basename.npz
# outputs: blosum matrix, basename_blosum.npz; grid search results stored as dictionary of open/extend params: list of lddts for val pairs,  basename_lddt_grid.pkl; param dictionary with k, lam basename_karlin_params.pkl


def parse_arguments():
    if len(sys.argv) != 2:
        print("Usage: compute_BLOSUM_gaps_lam_k.py <path_to_one_hot_dictionary.npz>")
    return sys.argv[1]


if __name__ == "__main__":
    
    oh_path = parse_arguments() 
    
    print(jax.devices())
    
    # base name (strip off .npz)
    basename = oh_path[:-4]
 
    # hardcoded paths
    data_path = "/cluster/tufts/pettilab/shared/structure_comparison_data"
    train_pairs_path = f"{data_path}/protein_data/pairs_training.csv"
    train_list_path = f"{data_path}/protein_data/train.csv"
    coord_path = f"{data_path}/protein_data/allCACoord.npz"
    val_path = f"{data_path}/protein_data/given_validation_alignments.npz"
    lddts_given_path =f"{data_path}/protein_data/pairs_validation_lddts.csv"
    
    
    # compute blosum and save, unless is 3Di or aa
    if basename not in ["3Di","aa"]:
        print("COMPUTING BLOSUM....")
        blosum = compute_blosum(oh_path, train_pairs_path, batch_size = 1000, save_path = basename+"_blosum.npy")
        print("saved blosum matrix at {basename}_blosum.py")
    
    # compute lambda and k
    print("COMPUTING LAMBDA AND K....")
    lam, k = compute_lambda_k(oh_path, basename+"_blosum.npy", train_list_path)
    print(f"lambda = {lam}, k = {k}")
          
          
    # do grid search (this will save lddt grid)
    print("RUNNING GRID SEARCH....")
    lddt_d = run_grid_search(oh_path, basename+"_blosum.npy", coord_path, val_path, lddts_given_path, save_path=basename+"_lddt_grid.pkl")
        
    # save parameters in dictionary
    params = {}
    params["lam"] = lam
    params["k"]=k

    
    pickle.dump(params, open(basename+"_karlin_params.pkl", "wb"))
