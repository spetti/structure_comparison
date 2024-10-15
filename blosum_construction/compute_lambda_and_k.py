import ctypes
import numpy as np
import csv
from matplotlib import pyplot as plt
import sys
import os


# example usage 
# data_path="/cluster/tufts/pettilab/shared/structure_comparison_data"
#  python compute_lambda_and_k.py $data_path/alphabets_blosum_coordinates/3Di/3Di.npz $data_path/alphabets_blosum_coordinates/3Di/3Di.npy $data_path/train_test_val/train.csv


def parse_arguments():
    if len(sys.argv) != 4:
        print("Usage: script.py <oh_path> <blosum_path> <training_names>")
        sys.exit(1)

    oh_path = sys.argv[1]
    blosum_path = sys.argv[2]
    train_path = sys.argv[3]

    return oh_path, blosum_path, train_path

# Load the shared library
lib = ctypes.CDLL('/cluster/tufts/pettilab/spetti04/structure_comparison/blosum_construction/psublast/libkarlin.so')  # For Windows use '.dll' instead of '.so'



# Define the C function prototype
lib.karlin.argtypes = [
    ctypes.c_int,                   # low (int)
    ctypes.c_int,                   # high (int)
    ctypes.POINTER(ctypes.c_double),# pr (double *)
    ctypes.POINTER(ctypes.c_double),# lambda (double *)
    ctypes.POINTER(ctypes.c_double) # K (double *)
]

lib.karlin.restype = ctypes.c_int    # Return type int

# Python function to call the C function
def call_karlin(low, high, pr_array):
    # Convert pr_array to a ctypes array
    pr = (ctypes.c_double * len(pr_array))(*pr_array)
    
    # Prepare ctypes variables to hold lambda and K
    lambda_val = ctypes.c_double()
    K_val = ctypes.c_double()
    
    # Call the C function
    result = lib.karlin(low, high, pr, ctypes.byref(lambda_val), ctypes.byref(K_val))
    
    if result != 1:
        raise RuntimeError("C function failed")
    
    # Return the computed lambda and K
    return lambda_val.value, K_val.value

def compute_lambda_k(oh_path,blosum_path,train_path):
    oh_d = np.load(oh_path)
    blosum = np.load(blosum_path).astype(int)
    training_names = [row[0] for row in csv.reader(open(train_path, newline=''))]
    
    A = blosum.shape[0]
    # compute background probs of each letter
    counts = np.zeros(A)
    for name in training_names:
        try:
            counts += np.sum(oh_d[name], axis = 0)
        except:
            print(f"skipping {name}, not in oh_d")
    background_probs = counts/np.sum(counts)
            
    # compute background probs of each score; store in vector    
    low = np.min(blosum)
    high = np.max(blosum)
    probs = np.zeros(high-low +1)
    for i in range(A):
        for j in range(A):
            probs[blosum[i,j]-low] += background_probs[i]*background_probs[j]
       
    # call Karlin code
    lam, k = call_karlin(low, high, probs)
    
    return lam, k



# RUN IT
if __name__ == "__main__":
    oh_path, blosum_path, train_path = parse_arguments()
    oh_d = np.load(oh_path)
    blosum = np.load(blosum_path).astype(int)
    training_names = [row[0] for row in csv.reader(open(train_path, newline=''))]
    
    A = blosum.shape[0]
    # compute background probs of each letter
    counts = np.zeros(A)
    for name in training_names:
        try:
            counts += np.sum(oh_d[name], axis = 0)
        except:
            print(f"skipping {name}, not in oh_d")
    background_probs = counts/np.sum(counts)
            
    # compute background probs of each score; store in vector    
    low = np.min(blosum)
    high = np.max(blosum)
    probs = np.zeros(high-low +1)
    for i in range(A):
        for j in range(A):
            probs[blosum[i,j]-low] += background_probs[i]*background_probs[j]
       
    # call Karlin code
    lam, k = call_karlin(low, high, probs)
    print(lam, k)
