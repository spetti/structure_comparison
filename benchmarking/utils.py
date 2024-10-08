import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import argparse
import sys
import os
import csv


########## FUNCTIONS FOR COMPUTING SIMILARITY MATRICES, ALIGNING, AND LDDT ################

def pad_and_stack(matrices, pad_value=0):
    # pad to smallest power of 2 greater than the length of the first dimension
    max_len = max(matrix.shape[0] for matrix in matrices)
    pad_to = int(jnp.where(max_len < 1, 1, 2 ** jnp.ceil(jnp.log2(max_len))))
    #print(int(pad_to))
    
    # Get the original lengths of each matrix
    original_lengths = [matrix.shape[0] for matrix in matrices]
    
    # Pad each matrix to have the same first dimension length (max_len)
    padded_matrices = [
        jnp.pad(matrix, ((0, pad_to - matrix.shape[0]), (0, 0)), constant_values=pad_value)
        for matrix in matrices
    ]
    
    # Stack the padded matrices along a new axis
    stacked_tensor = jnp.stack(padded_matrices, axis=0)
    
    return stacked_tensor, original_lengths

def pad_and_stack_manual(matrices, pad_to, pad_value=0):
    # pad to smallest power of 2 greater than the length of the first dimension
    max_len = max(matrix.shape[0] for matrix in matrices)
    
    # Get the original lengths of each matrix
    original_lengths = [matrix.shape[0] for matrix in matrices]
    
    # Pad each matrix to have the same first dimension length (max_len)
    padded_matrices = [
        jnp.pad(matrix, ((0, pad_to - matrix.shape[0]), (0, 0)), constant_values=pad_value)
        for matrix in matrices
    ]
    
    # Stack the padded matrices along a new axis
    stacked_tensor = jnp.stack(padded_matrices, axis=0)
    
    return stacked_tensor, original_lengths


def sim_mtx(oh_seq1, oh_seq2, blosum):
    return jnp.einsum('ij,jk,lk->il', oh_seq1, blosum, oh_seq2) 

v_sim_mtx = jax.jit(jax.vmap(sim_mtx, in_axes= (None, 0, None)))
vv_sim_mtx = jax.jit(jax.vmap(sim_mtx, in_axes= (0, 0, None)))


def sw_affine(restrict_turns=True, 
             penalize_turns=True,
             batch=True, unroll=2, NINF=-1e30):
  """smith-waterman (local alignment) with affine gap"""
  # rotate matrix for vectorized dynamic-programming
  

  def rotate(x):   
    # solution from jake vanderplas (thanks!)
    a,b = x.shape
    ar,br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
    i,j = (br-ar)+(a-1),(ar+br)//2
    n,m = (a+b-1),(a+b)//2
    output = {"x":jnp.full([n,m],NINF).at[i,j].set(x), "o":(jnp.arange(n)+a%2)%2}
    return output, (jnp.full((m,3), NINF), jnp.full((m,3), NINF)), (i,j)

  # fill the scoring matrix
  def sco(x, lengths, gap=0.0, open=0.0, temp=1.0):

    def _soft_maximum(x, axis=None, mask=None):
      def _logsumexp(y):
        y = jnp.maximum(y,NINF)
        if mask is None: return jax.nn.logsumexp(y, axis=axis)
        else: return y.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(y - y.max(axis, keepdims=True)), axis=axis))
      return temp*_logsumexp(x/temp)

    def _cond(cond, true, false): return cond*true + (1-cond)*false
    def _pad(x,shape): return jnp.pad(x,shape,constant_values=(NINF,NINF))
      
    def _step(prev, sm): 
      h2,h1 = prev   # previous two rows of scoring (hij) mtxs

      Align = jnp.pad(h2,[[0,0],[0,1]]) + sm["x"][:,None]
      Right = _cond(sm["o"], _pad(h1[:-1],([1,0],[0,0])),h1)
      Down  = _cond(sm["o"], h1,_pad(h1[1:],([0,1],[0,0])))

      # add gap penalty
      if penalize_turns:
        Right += jnp.stack([open,gap,open])
        Down += jnp.stack([open,open,gap])
      else:
        gap_pen = jnp.stack([open,gap,gap])
        Right += gap_pen
        Down += gap_pen

      if restrict_turns: Right = Right[:,:2]
      
      h0_Align = _soft_maximum(Align,-1)
      h0_Right = _soft_maximum(Right,-1)
      h0_Down = _soft_maximum(Down,-1)
      h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1)
      return (h1,h0),h0

    # mask
    a,b = x.shape
    real_a, real_b = lengths
    mask = (jnp.arange(a) < real_a)[:,None] * (jnp.arange(b) < real_b)[None,:]
    x = x + NINF * (1 - mask)

    sm, prev, idx = rotate(x[:-1,:-1])
    hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]

    # sink
    return _soft_maximum(hij + x[1:,1:,None], mask=mask[1:,1:,None])

  # traceback to get alignment (aka. get marginals)
  traceback = jax.grad(sco)

  # add batch dimension
  if batch: return jax.vmap(traceback,(0,0,None,None,None))
  else: return traceback

v_aln_w_sw = jax.jit(sw_affine(batch=True))


def pw(x):
    return jnp.sqrt(1e-10 + jnp.sum(
      (x[:, None] - x[None, :])**2, axis=-1))

def lddt2(coord_1, coord_2, aln, n, query_length):
    
    # Compute pw distances; however many are under 15 in structure 1 is the denom
    pw_1= pw(coord_1)
    mask = (jnp.arange(pw_1.shape[0]) < query_length)[:,None] * (jnp.arange(pw_1.shape[0]) < query_length)[None,:]
    denom =4*(jnp.sum(jnp.where(pw_1 <15,1,0)*mask)- query_length)/2
    
    # Reduce so that only aligned positions appear first
    row_indices, col_indices = jnp.where(aln > 0.95, size = pw_1.shape[0], fill_value =0)
    reduced_coord_1 = jnp.zeros_like(coord_1)
    reduced_coord_1 = reduced_coord_1.at[:len(row_indices)].set(jnp.take(coord_1, row_indices, axis=0))
    reduced_coord_2 = jnp.zeros_like(coord_2)
    reduced_coord_2 = reduced_coord_2.at[:len(col_indices)].set(jnp.take(coord_2, col_indices, axis=0))

    # Compute distance differences
    pw_1 = pw(reduced_coord_1)
    pw_2 = pw(reduced_coord_2)
    
    # Mask restricts to pairs we care about; n is number of aligned positions and we have permuted so these are the first n
    mask = (jnp.arange(pw_1.shape[0]) < n)[:,None] * (jnp.arange(pw_1.shape[0]) < n)[None,:]
    mask *= jnp.triu(jnp.ones_like(pw_1), k=1)
    mask *= (pw_1<15)

    # Count how many under thresholds
    distance_diffs = jnp.abs(pw_1-pw_2)

    num = jnp.sum((distance_diffs<0.5)*mask) + jnp.sum((distance_diffs<1.0)*mask) + jnp.sum((distance_diffs<2.0)*mask) + jnp.sum((distance_diffs<4.0)*mask) 

    #return num/denom, unless denom is zero
    return jnp.where(denom != 0, num / denom, 0)

v_lddt= jax.jit(jax.vmap(lddt2,in_axes= (None, 0, 0,0, None))) 
vv_lddt= jax.jit(jax.vmap(lddt2,in_axes= (0, 0, 0,0, 0))) 



def sim_mtx_blurry(nhot_1, nhot_2, tMtx):
    
    #compute blurry vectors 
    A =jnp.einsum('ik,kj->ij', nhot_1, tMtx)[:,:-1]
    B =jnp.einsum('ik,kj->ij', nhot_2, tMtx)[:,:-1]

    # Expand A and B to 3D arrays for broadcasting, aligning them for element-wise minimum calculation
    A_expanded = A[:, jnp.newaxis, :]  # Shape becomes (A_rows, 1, A_columns)
    B_expanded = B[jnp.newaxis, :, :]  # Shape becomes (1, B_columns, B_rows)
    
    # Calculate the minimum for each pair of elements from A and B
    min_elements = jnp.minimum(A_expanded, B_expanded)
    max_elements = jnp.maximum(A_expanded,B_expanded)
    
    # Sum over the last dimension to get the final result
    result = jnp.sum(min_elements, axis=2)/jnp.where(jnp.sum(max_elements, axis=2)==0, -1,jnp.sum(max_elements, axis=2)) # Shape is rows_A, rows_B, 1001
    return result
    
v_sim_mtx_blurry = jax.jit(jax.vmap(sim_mtx_blurry, in_axes= (None, 0, None)))
vv_sim_mtx_blurry = jax.jit(jax.vmap(sim_mtx_blurry, in_axes= (0, 0, None)))




########## FUNCTIONS TO CHECK KEY AND SEQUENCE LENGTH COMPATIBILITY ################

# returns protein_name: length based on some input dictionary with values of shape Lx *
def make_name_to_length_d(oh_d):
    name_to_length_d ={}
    for key in oh_d.keys():
        name_to_length_d[key]= oh_d[key].shape[0]
    return name_to_length_d

# returns keys that are not in both dictionaries
def check_keys(oh_d, coord_d):
    match = oh_d.keys()==coord_d.keys()
    if match:
        print("all keys match")
        return []
    else:
        b1 = oh_d.keys()-coord_d.keys()
        b2 = coord_d.keys()-oh_d.keys()
        print("in hot_d but not coord_d:")
        print(b1)
        print("in coord_d but not hot_d:")
        print(b2)
        return list(b1) + list(b2)

# checks whether proteins in the common set of keys are reported to be the same in each dictionary
def check_lengths(d1,d2):
    bad_list = []
    for key in set(d1.keys()):
        l1 = d1[key]
        l2 = d2.get(key,-1)
        if l2 >0:
            if l1!= l2:
                print(f"lengths of {key} differ: {l1},{l2}")
                bad_list.append(key)
    return bad_list

def check_keys_and_lengths(hot_d, coord_d):
    bad_list = check_keys(hot_d, coord_d)
    bad_list+=check_lengths(make_name_to_length_d(hot_d),make_name_to_length_d(coord_d))
    return bad_list
              
