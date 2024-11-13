# nH_alphabetizer.py
# J. Cool
# Revised 10-01-2024

# Usage: python nH_alphabetizer.py <weights.pt> <to_be_alphabetized.pkl> [--outfile_name STR] [--batch_size INT] [n_workers INT]
# Input: model.pt (from nH_vqvae.py), to_be_alphabetized.pkl (dictionary of protein names and Lx1001-dim nHot data for each protein)
# Output: a .npz, a dictionary with the same keys as to_be_alphabetized, but with the values replaced by the corresponding embeddings indices as one-hot vectors

import argparse
from pathlib import Path
import pickle
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Encoder
class Encoder(nn.Module):
    def __init__(self, hidden_channels, embedding_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1000, hidden_channels),
            nn.BatchNorm1d(hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# Quantizer
# we only need the forward pass, and we only need the indices... hence a paired-down Encoder
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        distances = (torch.sum(x**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(x, self.embedding.weight.t()))        

        codebook_index = torch.argmin(distances, dim=1)
        codebook_index_one_hot = torch.nn.functional.one_hot(codebook_index, num_classes=self.num_embeddings)
        return codebook_index_one_hot
    
class VQVAE(nn.Module):
    def __init__(self, hidden_channels, embedding_dim, num_embeddings):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(hidden_channels, embedding_dim)
        self.VQ = VectorQuantizer(num_embeddings, embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.VQ(z)

def load_query_protein(filepath):
    assert filepath.endswith('.pkl'), f'Input file must be .pkl'

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    tensor_dict = {}

    for key, value in data.items():
        truncated_value = value[:, :-1]
        tensor_value = torch.tensor(truncated_value, dtype=torch.float32).unsqueeze(1)
        tensor_dict[key] = tensor_value
    
    return tensor_dict

# Loads the .pt model produced by nH_vqvae.py
def load_model(model_path, device):
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)

    # Extract the 'VQ' state dict
    vq_state_dict = state_dict['VQ']
    num_embeddings = vq_state_dict['embedding.weight'].size(0)
    print(f"Num Clusters: {num_embeddings}")
    
    # Now initialize the model with the correct number of embeddings
    model = VQVAE(hidden_channels=1000, embedding_dim=2, num_embeddings=num_embeddings).to(device)
    
    # Load the rest of the state dict into the model
    model.encoder.load_state_dict(state_dict['encoder'], strict=True)
    model.VQ.load_state_dict(state_dict['VQ'], strict=True)

    model.eval()

    return model

def get_embedding_indices_for_protein(protein_tensor, model, device):
    with torch.no_grad():
        protein_tensor = protein_tensor.to(device)
        embedding_indices = model(protein_tensor[:, 0, :]) # reshaping might be superfluous...
 
    return embedding_indices.cpu().numpy()

# Returns a dictionary with the same keys as data, but with the values replaced by the corresponding embeddings indices
def alphabetize(model, data, device, batch_size, num_workers):
    embeddings_dict = {}
    
    for key, protein_tensor in data.items():
        dataset = TensorDataset(protein_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == 'cuda'))

        embeddings_list = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(device)
                batch_embeddings = get_embedding_indices_for_protein(batch, model, device)
                embeddings_list.append(batch_embeddings)

        embeddings = np.concatenate(embeddings_list, axis=0)
        embeddings_dict[key] = embeddings

    return embeddings_dict

def save(alphabetized_dict, outfile_name):
    np.savez(outfile_name, **alphabetized_dict)

# ----- MAIN -----

def main():
    parser = argparse.ArgumentParser(description='Alphabetize n-hot vectors using weights.')

    parser.add_argument('weights', type=str, help='Path to the model weights (.pt)')
    parser.add_argument('query', type=str, help='Path to the dictionary of proteins to be alphabetized (.pkl)')

    parser.add_argument('--batch_size', type=int, default=512, help='Batch size (default: 512)')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of workers (default: 0)')
    parser.add_argument('--outfile_name', type=str, help='Optional output file name')
    args = parser.parse_args()

    start_time = time.time()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    weights = args.weights
    query = args.query
    batch_size = args.batch_size
    n_workers = args.n_workers
    outfile_name = args.outfile_name or f'./alphabetized/{Path(query).stem}_alphabetized_with_{Path(weights).stem}.npz'

    data = load_query_protein(query) # must be preprocessed as .pt
    model = load_model(weights, device)

    alphabetized = alphabetize(model, data, device, batch_size, n_workers)
    save(alphabetized, outfile_name)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'Alphabetized {len(data)} inputs in {elapsed_time:.2f} seconds to {outfile_name}')

if __name__ == '__main__':
    main()
