import numpy as np
import jax
import jax.numpy as jnp
import csv
import sys
import os

# Add the path to the directory containing the file
sys.path.append(os.path.abspath('../benchmarking/'))
from utils import *


def parse_arguments():
    if len(sys.argv) != 4:
        print("Usage: script.py <oh_path> <training_pairs_path> <output_path>")
        sys.exit(1)

    oh_path = sys.argv[1]
    train_path = sys.argv[2]
    save_path = sys.argv[3]

    return oh_path, train_path,save_path


def get_pairs_and_alns(oh_path, pairs_path):
    oh_d = np.load(oh_path)

    # remove anything with 'd1e25a_' since we seem to be having some length issues
    bad_list = []
    bad_list.append('d1e25a_')

    pairs = []
    alns_as_lists = {}
    first = True
    n2l_d = make_name_to_length_d(oh_d)
    with open(pairs_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            pair = (row[1],row[2])
            if first:
                first = False
                continue  
            elif pair[0] in bad_list or pair[1] in bad_list:
                continue
            elif pair[0] not in n2l_d.keys():
                print(f"skipping {pair} because {pair[0]} is not in the one hot-d")
            elif pair[1] not in n2l_d.keys():
                print(f"skipping {pair} because {pair[1]} is not in the one hot-d")
            else:
                try:
                    alns_as_lists[pair]=[int(i) for i in row[-1].strip('[]').split()]
                except:
                    print(f"skipping {pair} because issue with alignment")
                    continue
                pairs.append(pair)
    
    # sort for better batching
    pair_max_length_pairs = [(pair, max(n2l_d[pair[0]], n2l_d[pair[1]])) for pair in pairs]
    sorted_keys = sorted(pair_max_length_pairs, key=lambda x: x[1])
    sorted_pairs = [key for key, shape in sorted_keys]
    pairs = sorted_pairs
    
    return pairs, oh_d, alns_as_lists, n2l_d

def compute_pairwise_counts(oh1, oh2, aln_list):
    
    aln_pos1 = jnp.where(aln_list > -.0001, size = oh1.shape[0], fill_value =oh1.shape[0]+oh2.shape[0]+1)[0]
    aln_pos2 = aln_list.at[aln_pos1].get(mode = "drop", fill_value =oh1.shape[0]+oh2.shape[0]+1)

    # reduce to matched positions
    reduced_oh1 = jnp.zeros_like(oh1)
    reduced_oh1 = reduced_oh1.at[:oh1.shape[0]].set(jnp.take(oh1, aln_pos1, axis=0, mode = "fill", fill_value = 0), mode = "drop")
    reduced_oh2 = jnp.zeros_like(oh2)
    reduced_oh2 = reduced_oh2.at[:oh2.shape[0]].set(jnp.take(oh2, aln_pos2, axis=0, mode = "fill", fill_value = 0), mode = "drop")
   
    # compute counts
    return jnp.einsum('mi,mj->ij', reduced_oh1, reduced_oh2)

vv_compute_pairwise_counts = jax.jit(jax.vmap(compute_pairwise_counts,in_axes= (0, 0, 0)))



def counts_to_blosum(counts):
    single_counts = np.sum(counts, axis = 0) + np.sum(counts, axis = 1)
    total_pairs = np.sum(counts)
    b = np.zeros_like(counts)
    
    # compute blosum score where there are positive counts
    for i in range(counts.shape[0]):
        for j in range(counts.shape[0]):
            if i == j:
                cij = counts[i,i]
            else:
                cij = counts[i,j]+counts[j,i]
            if cij != 0:
                # (cij/total_pairs) / (single_counts[i]*single_counts[j]/(2*total_pairs * 2*total_pairs))
                # = 4 cij* total_pairs / (single_counts[i]*single_counts[j])
                b[i,j]= np.round(  2/np.log(2)  * ( np.log(4*cij*total_pairs)-np.log( (1+(i!=j))*single_counts[i]*single_counts[j])  ))
                b[j,i]=b[i,j]
                
               
    # fill in zero count positions with lowest value
    lowest = np.min(b)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[0]):
            if i == j:
                cij = counts[i,i]
            else:
                cij = counts[i,j]+counts[j,i]
            if cij == 0:
                b[i,j] = lowest
                b[j,i] = lowest
    return b


def run_in_batches(long_list,oh_d,  alns_as_lists,n2l_d, batch_size, verbose = True):
    result = None
    for i in range(0, len(long_list), batch_size):
        batch = long_list[i:i + batch_size]  # Get the current batch
        counts = jnp.sum(run_batch(batch,oh_d,  alns_as_lists, n2l_d, verbose), axis = 0)
        if i == 0:
            result = counts
        else:
            result+= counts   # Process and extend results
        if verbose: print(f"finished batch {i}")
    return result

def run_batch(pairs, oh_d, alns_as_lists, n2l_d, verbose = True):

    # compute max length of any protein
    names=[item for tup in pairs for item in tup]
    max_len = max([n2l_d[name] for name in names])
    pad_to = int(jnp.where(max_len < 1, 1, 2 ** jnp.ceil(jnp.log2(max_len))))
    if verbose: print(pad_to)    
    
    alns = []
    for pair in pairs:
        alnL=alns_as_lists[pair]
        alns.append(jnp.pad(jnp.array(alnL),(0,pad_to-len(alnL)),constant_values =-1))
    alns = jnp.stack(alns)
        
    # get oh and coords for left and right part of pairs, padded
    lefts, _ = pad_and_stack_manual([oh_d[pair[0]] for pair in pairs],pad_to = pad_to)
    rights, _ = pad_and_stack_manual([oh_d[pair[1]] for pair in pairs], pad_to = pad_to)
    
    return vv_compute_pairwise_counts(lefts, rights, alns)


def compute_blosum(oh_path, pairs_path, batch_size = 5000, save_path = None):
    
    jax.config.update("jax_enable_x64", True)
    pairs,oh_d, alns_as_lists, n2l_d = get_pairs_and_alns(oh_path, pairs_path)
    counts = run_in_batches(pairs, oh_d, alns_as_lists, n2l_d, batch_size)
    blosum = counts_to_blosum(counts)
    print(blosum)
    if save_path:
        np.save(save_path, blosum)
    
    return blosum
    
    
def compute_counts(oh_path, pairs_path, batch_size = 5000, save_path = None):
    
    jax.config.update("jax_enable_x64", True)
    pairs,oh_d, alns_as_lists, n2l_d = get_pairs_and_alns(oh_path, pairs_path)
    counts = run_in_batches(pairs, oh_d, alns_as_lists, n2l_d, batch_size)
    return counts
    
# RUN IT
if __name__ == "__main__":
    oh_path, train_path, save_path = parse_arguments()    
    compute_blosum(oh_path, train_path, save_path=save_path)
    