# nH_alphabetizer.py
# J. Cool

# Usage: python nH_alphabetizer.py <weights.pt> <to_be_alphabetized.pkl> [batch_size (default 512)] [n_workers]
# Input: model.pt (from nH_vqvae.py), to_be_alphabetized.pkl (dictionary of protein names and Lx1001-dim nHot data for each protein)
# Output: .pkl, a dictionary with the same keys as to_be_alphabetized, but with the values replaced by the corresponding embeddings indices
# also– a .fasta, with the same keys as to_be_alphabetized, but with the values replaced by the corresponding embeddings characters

# TODO: restructure output to .npz, a dict of np matrices with 1-Hot embedding representations

import sys
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
        return codebook_index # Lx1
    
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
def load_weights(model_path, device):
    model = VQVAE(hidden_channels=1000, embedding_dim=2, num_embeddings=20).to(device)
    state_dict = torch.load(model_path, map_location=device)
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

def save(embedding_dict, outfile_name):
    with open(outfile_name, 'wb') as f:
        pickle.dump(embedding_dict, f)

# Returns a dictionary with the same keys as data, but with the values replaced by the corresponding embeddings characters
# TODO: terrible style
def alphabetize_fasta(model, data, device, batch_size, num_workers):
    CHARACTERS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'] # AA alphabet

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

        embedding_indices = np.concatenate(embeddings_list, axis=0)

        # Convert to characters
        embeddings = [CHARACTERS[i] for i in embedding_indices]
        embeddings_dict[key] = embeddings

    return embeddings_dict

def save_fasta(embedding_dict, outfile_name):
    with open(outfile_name, 'w') as f:
        for key, embeddings in embedding_dict.items():
            f.write(f'>{key}\n')
            embedding_str = ''.join(map(str, (item for sublist in embeddings for item in sublist)))
            f.write(f'{embedding_str}\n')

# ----- MAIN -----

def main():
    # TODO: better data validation, assertions, etc (own function)

    start_time = time.time()

    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print('Usage: python nH_alphabetizer.py <weights.pt> <to_be_alphabetized.pt> [batch_size (default 512)] [n_workers (default 0)]')
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    weights = sys.argv[1]
    query = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) >= 4 else 128
    n_workers = int(sys.argv[4]) if len(sys.argv) == 5 else 0 # terrible practice
    outfile_name = f'./alphabetized/{Path(query).stem}_alphabetized_with_{Path(weights).stem}.pkl'
    fasta_name = f'./alphabetized/{Path(query).stem}_alphabetized_with_{Path(weights).stem}.fasta'

    data = load_query_protein(query) # must be preprocessed as .pt
    model = load_weights(weights, device)

    reg_embeddings = alphabetize(model, data, device, batch_size, n_workers)
    save(reg_embeddings, outfile_name)

    fasta_embeddings = alphabetize_fasta(model, data, device, batch_size, n_workers)
    save_fasta(fasta_embeddings, fasta_name)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Alphabetized {len(data)} inputs in {elapsed_time:.2f} seconds to {outfile_name} and {fasta_name}')

if __name__ == '__main__':
    main()
