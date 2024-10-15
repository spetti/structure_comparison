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

# python query_list_by_all.py $data/alphabets_blosum_coordinates/allCACoord.npz $data/small_test_queries_by_10/query_list_1.csv  $data/alphabets_blosum_coordinates/MSE/MSE.npy $data/alphabets_blosum_coordinates/MSE/MSE.npz --out_location test_results --lam 0.318 --k 0.269 

#0.3178780674934387 0.2689192327152678

#python query_list_by_all.py $data/allCACoord.npz $data/small_test_queries_by_10/query_list_1.csv ./data/nH_mat.npy ./data/nH_oh.npz --blosum2_path ./data/nH_mat.npy --oh2_path ./data/nH_oh.npz --w1 1.0 --w2 1.0 --gap_open -10 --gap_extend -2 --out_location test_results_3Di_3Dn

#python query_list_by_all.py $data/alphabets_blosum_coordinates/allCACoord.npz $data/small_test_queries_by_10/query_list_1.csv $data/alphabets_blosum_coordinates/transition_mtx.npy $data/alphabets_blosum_coordinates/blurry_vecs.npz --jaccard_blosum_list $data/alphabets_blosum_coordinates/jaccard_blosum.npy --out_location test_results_BV

# functions 


def run_in_batches(query, long_list, batch_size, params):
    lddts = []
    scores = []
    for i in range(0, len(long_list), batch_size):
        batch = long_list[i:i + batch_size]  # Get the current batch
        ls, ss = run_batch(query,batch, params)
        lddts.extend(ls)
        scores.extend(ss)
        #print(f"finished batch {i}")
    return lddts, scores

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
    if params["use_two"]:
        padded_query_oh2 = jnp.pad(oh_d2[query], ((0, params["query_pad_to"] - query_length),(0,0)), mode='constant') 
    padded_query_coordinates = jnp.pad(coord_d[query], ((0, params["query_pad_to"] - query_length),(0,0)), mode='constant') 
    
    # make similarity matrices
    if params["blurry"]:
        sim_tensor = v_sim_mtx_blurry(padded_query_oh, oh_db, blosum)
        sim_tensor = v_replace_jaccard_w_blosum_score(sim_tensor, params["jaccard_blosum_list"])
    elif params["use_two"]:
        sim_tensor1 = v_sim_mtx(padded_query_oh, oh_db, blosum)
        sim_tensor2= v_sim_mtx(padded_query_oh2, oh_db2, blosum2)
        sim_tensor = params["w1"]* sim_tensor1 + params["w2"]*sim_tensor2
    else:
        sim_tensor = v_sim_mtx(padded_query_oh, oh_db, blosum)

    # align (gap, open, temp)
    length_pairs = jnp.column_stack((jnp.full((len(lengths),), query_length), jnp.array(lengths)))
    aln_tensor = v_aln_w_sw(sim_tensor, length_pairs, params["gap_extend"], params["gap_open"],params["temp"])
    aln_tensor = (aln_tensor>params["soft_aln_thresh"]).astype(int)

    
    # compute bit scores
    # see https://www.ncbi.nlm.nih.gov/BLAST/tutorial/Altschul-1.html, eq 2
    if params["use_two"]:
            scores1 = vv_get_score(sim_tensor1, aln_tensor, length_pairs, params["gap_extend"], params["gap_open"])
            scores2 = vv_get_score(sim_tensor2, aln_tensor, length_pairs, params["gap_extend"], params["gap_open"])
            scores = params["w1"]*(params["lam"]*scores1- jnp.log(params["k"]))/jnp.log(2)+params["w2"]*(params["lam2"]*scores1- jnp.log(params["k2"]))/jnp.log(2)
    else:
        scores = vv_get_score(sim_tensor, aln_tensor, length_pairs, params["gap_extend"], params["gap_open"])
        scores = (params["lam"]*scores- jnp.log(params["k"]))/jnp.log(2)
    
    # compute lddts
    lddts = v_lddt(coord_d[query], db_coords, aln_tensor, jnp.sum((aln_tensor>0.95).astype(int), axis = [-2,-1]), query_length)
    
    return lddts, scores

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process input file paths and alignment parameters.")
    
    # Required positional arguments
    parser.add_argument('coord_path', type=str, help="Path to the coordinate file")
    parser.add_argument('query_list_path', type=str, help="Path to the query list file")
    parser.add_argument('blosum_path', type=str, help="Path to the BLOSUM matrix file or if blurry transition mtx file")
    parser.add_argument('oh_path', type=str, help="Path to the OH file or if blurry NH file")
    
    # Values need to convert scores into bitscores 
    parser.add_argument('--lam', type = float, default = None, help="lambda value for blosum2")
    parser.add_argument('--k', type = float, default = None, help="k value for blosum2")
    
    # Optional arguments for the second set of paths
    parser.add_argument('--blosum2_path', type=str, help="Path to the second BLOSUM matrix file (optional)")
    parser.add_argument('--oh2_path', type=str, help="Path to the second OH file (optional)")
    parser.add_argument('--w1', type=float, default=0.5, help="Weight for the first BLOSUM and OH files (default: 0.5)")
    parser.add_argument('--w2', type=float, default=0.5, help="Weight for the second BLOSUM and OH files (default: 0.5)")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size; lower will take more time, less memory")
    parser.add_argument('--jaccard_blosum_list', type=str, default = None, help="path to list of length 100 with BLOSUM values")
    parser.add_argument('--lam2', type = float, default = None, help="lambda value for blosum2")
    parser.add_argument('--k2', type = float, default = None, help="k value for blosum2")
    
    # Optional arguments with default values for gap penalties and output location
    parser.add_argument('--gap_open', type=float, default=-10, help="Gap opening penalty (default: -10)")
    parser.add_argument('--gap_extend', type=float, default=-2, help="Gap extension penalty (default: -2)")
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
    
    # Load relevant data and check dimensions of one example
    coord_d = np.load(args.coord_path)
    oh_d = np.load(args.oh_path)
    blosum = np.load(args.blosum_path).astype(float)
    query_list = load_csv_to_list(args.query_list_path)
    if oh_d[list(oh_d.keys())[0]].shape[1]!=blosum.shape[1]:
        raise ValueError(f"one-hot encoding length does not match blosum shape {blosum.shape[1]}")
    if args.blosum2_path and args.oh2_path:
        oh_d2 = np.load(args.oh2_path)
        blosum2 = np.load(args.blosum2_path).astype(float)
        if oh_d2[list(oh_d2.keys())[0]].shape[1]!=blosum2.shape[1]:
            raise ValueError(f"one-hot encoding length does not match blosum2 shape {blosum2.shape[1]}")

    # Set parameters
    params={}
    params["gap_open"] = args.gap_open
    params["gap_extend"] = args.gap_extend
    params["temp"] = 1e-3 # do not change
    params["soft_aln_thresh"] = 0.5 # do not change
    params["w1"]=args.w1
    params["w2"]=args.w2
    params["lam"]= args.lam
    params["k"]=args.k
    params["lam2"]=args.lam2
    params["k2"]=args.k2
    params["use_two"] = bool(args.blosum2_path)
    params["blurry"] = bool(args.jaccard_blosum_list)
    
    #check the length of jaccard blosum list; needs to be 100 now
    if params["blurry"]:
        params["jaccard_blosum_list"] = np.load(args.jaccard_blosum_list)+0.0
        if params["jaccard_blosum_list"].shape[0]!= 100:
            raise ValueError("jaccard BLOSUM list must have length 100")
     
    
    # get longest query_length
    max_query_length = max([oh_d[query].shape[0] for query in query_list])
    params["query_pad_to"] = int(max_query_length)
    print(f"padding queries to {max_query_length}")
    
    # Sort the database by length for better batching
    key_shape_pairs = [(key, oh_d[key].shape[0]) for key in oh_d.keys()]
    sorted_keys = sorted(key_shape_pairs, key=lambda x: x[1])
    sorted_names = [key for key, shape in sorted_keys]

    
    # Store results: make one output file for the entire query list
    if not os.path.exists(args.out_location):
        os.makedirs(args.out_location)
    filename = f"{os.path.basename(args.query_list_path)}.output"
    
    # Run one by all and write output results
    for query in query_list:
        # Compute one by all LDDTs
        print(query)
        #half the batch size if the query is longer than 512
        bs = args.batch_size
        if max_query_length >512:
            bs = int(bs/2)
        lddts, scores = run_in_batches(query, sorted_names,bs, params)

        prods = [lddts[i]*scores[i] for i in range(len(lddts))]
                                                                 
        # Sort by geo mean
        name_triples = [(sorted_names[i], prods[i], lddts[i],scores[i]) for i in range(len(prods))]
        sorted_quads = sorted(name_triples, key=lambda x: x[1], reverse = True)
    
        with open(f"{args.out_location}/{filename}", "a") as file:
            for quad in sorted_quads:
                file.write(f"{query} {quad[0]} {quad[1]} {quad[2]} {quad[3]}\n")
 
    









