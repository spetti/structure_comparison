# nH_vqvae.py
# J. Cool

# Given aligned pairs of 1000-dim nHot data, finds nonlinear clustering.

# Usage: python nH_vqvae.py <trainset.pt> <validationset.pt> [model_and_dict_name] [seed]
# Input: trainset.pt, validationset.pt (pairs of aligned nHot data) – [*, 2, 1000]
# Output: models/<model_and_dict_name>_model.pt, codebooks/<model_and_dict_name>_codebook.pkl (decoded embeddings)

# TODO: Fix deterministic seed...

import argparse
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_channels, embedding_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, 1000)
        )

    def forward(self, x):
        return self.decoder(x)

# Quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost
        
        # uniform initialization of embeddings in 2d space
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, x):
        distances = (torch.sum(x**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(x, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # closest embedding
        quantized = torch.matmul(encodings, self.embedding.weight)

        # Loss (e and q are the same– serve to move embeddings and encodings closer to each other via backprop)
        e_latent_loss = F.mse_loss(quantized.detach(), x) # ≥0
        q_latent_loss = F.mse_loss(quantized, x.detach()) # ≥0
        loss = q_latent_loss + (self.commitment_cost * e_latent_loss)

        quantized = x + (quantized - x).detach() # backpropagation passthru
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, encodings, perplexity, encoding_indices

# VQVAE
class VQVAE(nn.Module):
    def __init__(self, hidden_channels, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(hidden_channels, embedding_dim)
        self.VQ = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(hidden_channels, embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        vq_loss, quantized, encodings, perplexity, encoding_indices = self.VQ(z)
        predicted = self.decoder(quantized)
        return vq_loss, predicted, perplexity, encoding_indices

# Training loop
def train_vqvae(model, train_loader, n_epochs, optimizer, criterion, device):
    model.train()
    for i in range(n_epochs):
        total_loss = 0
        total_perplexity = 0
        num_batches = 0

        for feat_x, feat_y in train_loader:
            feat_x = feat_x.clone().detach().to(device)
            feat_y = feat_y.clone().detach().to(device)

            optimizer.zero_grad()

            vq_loss, predicted, perplexity, _ = model(feat_x)
            recon_loss = criterion(predicted, feat_y) # better for low numbers

            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        print(f'Epoch {i+1}, Loss: {avg_loss:.5f}, Perplexity: {avg_perplexity:.5f}')

# Evaluation loop
def evaluate_vqvae(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_perplexity = 0
    num_batches = 0

    with torch.no_grad():
        for feat_x, feat_y in test_loader:
            feat_x = feat_x.clone().detach().to(device)
            feat_y = feat_y.clone().detach().to(device)

            vq_loss, predicted, perplexity, _ = model(feat_x)
            recon_loss = criterion(predicted, feat_y)

            loss = recon_loss + vq_loss

            total_loss += loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        print(f'Test Loss: {avg_loss:.5f}, Test Perplexity: {avg_perplexity:.5f}')

# Save model weights for inference (encoder and VQ embeddings)
def save_for_inference(model, filename):
    model_name = f'./models/{Path(filename).stem}_model.pt'

    combined_state_dict = {
        'encoder': model.encoder.state_dict(), # encoder weights
        'VQ': model.VQ.state_dict() # VQ embeddings, IN ORDER
    }

    torch.save(combined_state_dict, model_name)
    print(f'Model saved to {model_name}')

# DEPRECIATED!
# TODO: make this optional
def save_decoded_codebook_vectors(model, device, filename):
    filename = f'./codebooks/{Path(filename).stem}_codebook.pkl'

    model.eval()
    with torch.no_grad():
        decoded_codebook_vectors = {} # decoded embeddings (our clusters)
        
        for i in range(model.VQ.num_embeddings):
            embedding_vector = model.VQ.embedding.weight[i].unsqueeze(0).to(device) # grab each embedding vector
            embedding = model.decoder(embedding_vector) # decoder it...

            embedding = embedding.cpu().numpy().flatten()
            
            key = i
            decoded_codebook_vectors[key] = embedding

    with open(filename, 'wb') as f:
        pickle.dump(decoded_codebook_vectors, f)
    print(f'Decoded codebook dictionary saved to {filename}')


def load_data(trainfile, validationfile, batch_size):
    assert trainfile.endswith('.pt'), f'Trainfile file must be .pt'
    assert validationfile.endswith('.pt'), f'Trainfile file must be .pt'

    traindata = torch.load(trainfile)
    validationdata = torch.load(validationfile)

    assert traindata.shape[1] == 2 and traindata.shape[2] == 1000, f'Train data must be of shape [*, 2, 1000], not {traindata.shape}'
    assert validationdata.shape[1] == 2 and validationdata.shape[2] == 1000, f'Validation data must be of shape [*, 2, 1000], not {validationdata.shape}'
    
    print(f'Trainset of {traindata.size(0)} and validationset of {validationdata.size(0)} pairs...')

    train_dataset = TensorDataset(traindata[:, 0, :], traindata[:, 1, :])
    validation_dataset = TensorDataset(validationdata[:, 0, :], validationdata[:, 1, :])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, validation_loader

def set_seed(seed):
    torch.manual_seed(seed)  # CPU seed
    torch.cuda.manual_seed_all(seed)  # CUDA seed
    torch.backends.cudnn.deterministic = True  # Deterministic CUDA...
    torch.backends.cudnn.benchmark = False  # Disable CUDA benchmark
    if torch.backends.mps.is_built():
        torch.mps.manual_seed(seed)  # MPS seed

# ----- FUNCTIONAL -----

# FIXED PARAMETERS (following Foldseek)
EMBEDDING_DIM = 2
NUM_EMBEDDINGS = 20

# Variable parameters
hidden_channels = 1000
commitment_cost = 0.25
learning_rate = 1e-3
batch_size = 512
num_epochs = 8

# Making this easier to run on Macbook vs Cluster...
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

model = VQVAE(hidden_channels, EMBEDDING_DIM, NUM_EMBEDDINGS, commitment_cost).to(device)
optimizer = optim.Adam(model.parameters(), learning_rate)
criterion = nn.MSELoss()

def main():
    parser = argparse.ArgumentParser(description='nH VQVAE script with optional model name and seed.')
    
    # Required
    parser.add_argument('trainfile', type=str, help='Path to the training set (.pt)')
    parser.add_argument('validationfile', type=str, help='Path to the validation set (.pt)')
    
    # Optional
    parser.add_argument('--model_and_dict_name', type=str, default=None, help='Model and dictionary name (optional, defaults to trainfile)')
    parser.add_argument('--seed', type=int, default=68, help='Random seed (optional, defaults to 68)')
    args = parser.parse_args()

    trainfile = args.trainfile
    validationfile = args.validationfile
    model_and_dict_name = args.model_and_dict_name if args.model_and_dict_name else trainfile
    seed = args.seed

    set_seed(seed)
    
    train_loader, validation_loader = load_data(trainfile, validationfile, batch_size)

    train_vqvae(model, train_loader, num_epochs, optimizer, criterion, device)
    evaluate_vqvae(model, validation_loader, criterion, device)

    save_for_inference(model, model_and_dict_name)
    save_decoded_codebook_vectors(model, device, model_and_dict_name)

if __name__ == '__main__':
    main()
