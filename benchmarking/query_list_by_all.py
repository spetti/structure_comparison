#!/usr/bin/env python
# coding: utf-8

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import argparse
import sys
import os
import csv
from utils import *

## example usage:

#export data=/cluster/tufts/pettilab/shared/structure_comparison_data

# python query_list_by_all.py $data/alphabets_blosum_coordinates/allCACoord.npz $data/train_test_val/short_test_query_list.csv $data/alphabets_blosum_coordinates/nH_mat.npy $data/alphabets_blosum_coordinates/nH_oh.npz --out_location test_results_3Dn

#python query_list_by_all.py ./data/allCACoord.npz ./data/short_test_query_list.csv ./data/nH_mat.npy ./data/nH_oh.npz --blosum2_path ./data/nH_mat.npy --oh2_path ./data/nH_oh.npz --w1 1.0 --w2 1.0 --gap_open -10 --gap_extend -2 --out_location test_results_3Di_3Dn

# functions 


def run_in_batches(query, long_list, batch_size, params):
    result = []
    for i in range(0, len(long_list), batch_size):
        batch = long_list[i:i + batch_size]  # Get the current batch
        result.extend(run_batch(query,batch, params))   # Process and extend results
        #print(f"finished batch {i}")
    return result

def run_batch(query, names, params):
    
    # one hot encode database(s), pad to a common length that is a power of 2    
    oh_db, lengths = pad_and_stack([oh_d[name] for name in names])
    if params["use_two"]:
        oh_db2, _ = pad_and_stack([oh_d2[name] for name in names])
    
    query_length = oh_d[query].shape[0]
 
    # get coordinates for everything in database
    db_coords, _ = pad_and_stack([coord_d[name] for name in names])
       
    # pad query to a power of 2
    padded_query_oh = jnp.pad(oh_d[query], ((0, params["query_pad_to"] - query_length),(0,0)), mode='constant') 
    padded_query_coordinates = jnp.pad(coord_d[query], ((0, params["query_pad_to"] - query_length),(0,0)), mode='constant') 
    
    # make similarity matrices
    sim_tensor = v_sim_mtx(padded_query_oh, oh_db, blosum)
    if params["use_two"]:
        sim_tensor *= params["w1"]
        sim_tensor += params["w2"]*v_sim_mtx(padded_query_oh, oh_db2, blosum2)
    
    # align (gap, open, temp)
    length_pairs = jnp.column_stack((jnp.full((len(lengths),), query_length), jnp.array(lengths)))
    aln_tensor = v_aln_w_sw(sim_tensor, length_pairs, params["gap_extend"], params["gap_open"],params["temp"])
    
    # compute lddts (for loop for now)
    lddts = v_lddt(coord_d[query], db_coords, aln_tensor, jnp.sum((aln_tensor>0.95).astype(int), axis = [-2,-1]), query_length)
    
    return lddts

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process input file paths and alignment parameters.")
    
    # Required positional arguments
    parser.add_argument('coord_path', type=str, help="Path to the coordinate file")
    parser.add_argument('query_list_path', type=str, help="Path to the query list file")
    parser.add_argument('blosum_path', type=str, help="Path to the BLOSUM matrix file")
    parser.add_argument('oh_path', type=str, help="Path to the OH file")
    
    # Optional arguments for the second set of paths
    parser.add_argument('--blosum2_path', type=str, help="Path to the second BLOSUM matrix file (optional)")
    parser.add_argument('--oh2_path', type=str, help="Path to the second OH file (optional)")
    parser.add_argument('--w1', type=float, default=0.5, help="Weight for the first BLOSUM and OH files (default: 0.5)")
    parser.add_argument('--w2', type=float, default=0.5, help="Weight for the second BLOSUM and OH files (default: 0.5)")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size; lower will take more time, less memory")

    
    # Optional arguments with default values for gap penalties and output location
    parser.add_argument('--gap_open', type=int, default=-10, help="Gap opening penalty (default: -10)")
    parser.add_argument('--gap_extend', type=int, default=-2, help="Gap extension penalty (default: -2)")
    parser.add_argument('--out_location', type=str, default="./results", help="specify where output files will go")

    # Parsing arguments
    args = parser.parse_args()

    # Custom logic: Either both --blosum2_path and --oh2_path must be provided, or neither
    if bool(args.blosum2_path) != bool(args.oh2_path):
        parser.error("You must either provide both --blosum2_path and --oh2_path, or omit both.")

    return args

def load_csv_to_list(file_path):
    with open(file_path, mode='r', newline='') as csvfile:
        return [item for row in csv.reader(csvfile) for item in row]
    

    
if __name__ == "__main__":
    # Parse the arguments
    args = parse_arguments()
    
    # Load relevant data
    coord_d = np.load(args.coord_path)
    oh_d = np.load(args.oh_path)
    blosum = np.load(args.blosum_path)[:-1,:-1].astype(float)
    query_list = load_csv_to_list(args.query_list_path)
    if args.blosum2_path and args.oh2_path:
        oh_d2 = np.load(args.oh2_path)
        blosum2 = np.load(args.blosum2_path)[:-1,:-1].astype(float)

    # Set parameters
    params={}
    params["gap_open"] = args.gap_open
    params["gap_extend"] = args.gap_extend
    params["temp"] = .01
    params["w1"]=args.w1
    params["w2"]=args.w2
    params["use_two"] = bool(args.blosum2_path)
    
    # get longest query_length
    max_query_length = max([oh_d[query].shape[0] for query in query_list])
    params["query_pad_to"] = int(max_query_length)
    print(f"padding queries to {max_query_length}")
    
    # Sort the database by length for better batching
    key_shape_pairs = [(key, oh_d[key].shape[0]) for key in oh_d.keys()]
    sorted_keys = sorted(key_shape_pairs, key=lambda x: x[1])
    sorted_names = [key for key, shape in sorted_keys]

    
    # Store results
    if not os.path.exists(args.out_location):
        os.makedirs(args.out_location)
    
    # Run one by all and write output results
    for query in query_list:
        # Compute one by all LDDTs
        print(query)
        lddts = run_in_batches(query, sorted_names,args.batch_size, params)

        # Sort by LDDT
        name_lddt_pairs = [(sorted_names[i], lddts[i]) for i in range(len(lddts))]
        sorted_pairs = sorted(name_lddt_pairs, key=lambda x: x[1], reverse = True)
        

        # Make output file for the query
        filename = f"{query}.txt"
        with open(f"{args.out_location}/{filename}", "w") as file:
            for pair in sorted_pairs:
                file.write(f"{query} {pair[0]} {pair[1]}\n")
 
    









