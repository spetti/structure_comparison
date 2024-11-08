import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import argparse
import sys
import os
import csv
import pickle
import scipy.stats as ss
from utils import *

# for one discrete alphabet 

# example usage
# export data=/cluster/tufts/pettilab/shared/structure_comparison_data
# python gap_search_via_aln_benchmark.py $data/alphabets/aa.npz $data/alphabets/aa_blosum.npy $data/protein_data/allCACoord.npz $data/protein_data/given_validation_alignments.npz $data/protein_data/pairs_validation_lddts.csv test_lddt_d.pkl

def parse_arguments():
    if len(sys.argv) != 7:
        print("Usage: gap_search_via_aln_benchmark.py <oh_path> <blosum_path> <coord_path> <validation_pairs_path> <lddts_given_path> <output_path>")
        sys.exit(1)
    oh_path = sys.argv[1]
    blosum_path = sys.argv[2]
    coord_path = sys.argv[3]
    val_path = sys.argv[4]
    lddts_given_path = sys.argv[5]
    save_path = sys.argv[6]

    return oh_path, blosum_path, coord_path, val_path, lddts_given_path, save_path



# check dimensions and organize data
def get_data(oh_path, blosum_path, coord_path, val_aln_path, given_lddt_path):
    
    coord_d = np.load(coord_path)
    oh_d = np.load(oh_path)
    blosum = np.load(blosum_path).astype(float)
    
    # check alphabet size matches blosum size
    if oh_d[list(oh_d.keys())[0]].shape[1]!=blosum.shape[1]:
        raise ValueError(f"one-hot encoding length does not match blosum shape {blosum1.shape[1]}")
    
    # make sure coordinates and one hots have the same length and same seqs are represented
    bad_list = check_keys_and_lengths(oh_d, coord_d)
    bad_list.append('d1e25a_')
    
    n2l_d = make_name_to_length_d(coord_d)

    # load in validation pairs, their alignments and lddt of them (precomputed in organize_val_and_train)
    val_aln_d = dict(np.load(val_aln_path))
    val_aln_d_new = {}
    for key, val in val_aln_d.items():
        val_aln_d_new[tuple(key.split(','))] = val

    val_aln_d = val_aln_d_new
    val_aln_d_new = {}
    given_lddt_d = {}

    # Open the CSV file for reading
    with open(given_lddt_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            a, b, value = row[0], row[1], float(row[2])  # Convert value to float
            given_lddt_d[(a, b)] = value
    
    # check that keys match
    check_keys(given_lddt_d, val_aln_d)
    
    # check that no pair includes a protein on the bad list
    for key in val_aln_d.keys():
        if key[0] in bad_list or key[1] in bad_list:
            raise ValueError(f"pair {key} is bad and should not be used")
            
    # sort pairs by length of longer protein
    pair_list = sorted(list(val_aln_d.keys())) 
    pair_max_length_pairs = [(pair, max(n2l_d[pair[0]], n2l_d[pair[1]])) for pair in pair_list]
    sorted_keys = sorted(pair_max_length_pairs, key=lambda x: (x[1],x[0][0],x[0][1]))
    sorted_pairs = [key for key, shape in sorted_keys]
    pairs = sorted_pairs
    given_lddt_list = [given_lddt_d[pair] for pair in pairs]
        
    data = {}
    data["oh_d"]= oh_d
    data["blosum"]= blosum
    data["coord_d"] = coord_d
    data["n2l_d"]=n2l_d
    data["pairs"] = pairs
    data["given_lddt_list"] =  given_lddt_list
    
    return data
    

def run_in_batches(params, data):
    long_list = data["pairs"]
    batch_size = params["batch_size"]
    result = []
    for i in range(0, len(long_list), batch_size):
        batch = long_list[i:i + batch_size]  # Get the current batch
        result.extend(run_batch(batch, params, data))   # Process and extend results
        print(f"finished batch, {i} done")
    return result

def run_batch(pairs, params, data):

    oh_d= data["oh_d"]
    blosum = data["blosum"]
    coord_d = data["coord_d"]
    n2l_d = data["n2l_d"]
    if params["use_two"]:
        oh_d2 = data["oh_d2"]
        blosum2 = data["blosum2"]

    # compute max length of any protein
    names=[item for tup in pairs for item in tup]
    max_len = max([n2l_d[name] for name in names])
    pad_to = int(jnp.where(max_len < 1, 1, 2 ** jnp.ceil(jnp.log2(max_len))))
    #print(pad_to)

    #print(max([oh_d[pair[0]].shape[0] for pair in pairs]))
    # get oh and coords for left and right part of pairs, padded
    lefts, left_lengths = pad_and_stack_manual([oh_d[pair[0]] for pair in pairs],pad_to = pad_to)
    rights, right_lengths = pad_and_stack_manual([oh_d[pair[1]] for pair in pairs], pad_to = pad_to)
    
    if params["use_two"]:
        lefts2, _ = pad_and_stack_manual([oh_d2[pair[0]] for pair in pairs],pad_to = pad_to)
        rights2, _ = pad_and_stack_manual([oh_d2[pair[1]] for pair in pairs], pad_to = pad_to)

    left_coords, _ = pad_and_stack_manual([coord_d[pair[0]] for pair in pairs],pad_to = pad_to)
    right_coords, _ = pad_and_stack_manual([coord_d[pair[1]] for pair in pairs], pad_to = pad_to)

    # make similarity matrices
    if params["blurry"]:
        sim_tensor = vv_sim_mtx_blurry(lefts, rights, blosum)
        sim_tensor = v_replace_jaccard_w_blosum_score(sim_tensor, params["jaccard_blosum"])
    else:
        sim_tensor = vv_sim_mtx(lefts, rights, blosum)
        if params["use_two"]:
            sim_tensor *= params["w1"]
            sim_tensor += params["w2"]*vv_sim_mtx(lefts2, rights2, blosum2)

    # align (gap, open, temp)
    length_pairs = jnp.column_stack([jnp.array(left_lengths), jnp.array(right_lengths)])
    aln_tensor = v_aln_w_sw(sim_tensor, length_pairs, params["gap_extend"], params["gap_open"], params["temp"])
    aln_tensor = (aln_tensor>params["soft_aln_thresh"]).astype(int)

    # compute lddts 
    lddts = vv_lddt(left_coords, right_coords, aln_tensor, jnp.sum(aln_tensor, axis = [-2,-1]), jnp.array(left_lengths))

    return lddts

def grid_search(data,open_choices=np.arange(-20,0,2), extend_choices=np.arange(-3,.3,0.5), save_path = None):
#def grid_search(data, open_choices=[-5,-10], extend_choices=[-2,-1], save_path = None):
    print(open_choices, extend_choices)
    params = {}
    params["gap_extend"] = None
    params["gap_open"] = None
    params["temp"] = 1e-3 # do not change
    params["soft_aln_thresh"]=.5 # do not change
    params["use_two"]= False
    params["w1"] = None
    params["w2"] = None
    params["blurry"] = None
    params["jaccard_blosum"] = None
    params["batch_size"] = 200 # make smaller if running out of mem
    
    lddt_d = {}

    for o in open_choices:
        for e in extend_choices:
            print(o,e)
            params["gap_extend"] = e
            params["gap_open"] = o
            lddt_d[(o,e)] = run_in_batches( params, data)
 
    for key, val in lddt_d.items():
        lddt_d[key]= [_.item() for _ in val]
        
    if save_path:
        pickle.dump(lddt_d, open(save_path, "wb"))
    
    return lddt_d


def run_grid_search(oh_path, blosum_path, coord_path, val_path, lddts_given_path, save_path=None):
    
    data = get_data(oh_path, blosum_path, coord_path, val_path, lddts_given_path)
    lddt_d = grid_search(data, save_path =save_path)
    return lddt_d


if __name__ == "__main__":
    oh_path, blosum_path, coord_path, val_path, lddts_given_path, save_path = parse_arguments()    
    run_grid_search(oh_path, blosum_path, coord_path, val_path, lddts_given_path, save_path)

































