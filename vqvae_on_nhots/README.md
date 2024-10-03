# VQVAE clustering method

We train on a set of aligned pairs, formatted as a tensor of shape N x 2 x 1000. From the first member of each pair we train the model to predict the second. Each input is encoded and mapped to a "codebook vector" in a lower-dimensional latent space (2 dimensional). The position of this codebook vector is learned in training. The codebook vector closest to the input is then decoded. We hope this method leads to a clustering of 20 states that minimize information loss.

## `nH_vqvae.py`

To train a model, use `python nH_vqvae.py <trainset.pt> <validationset.py> [model name] [seed]`, where `trainset.py` and `validationset.py` are tensors of the above dimension, `model name` is a string, and `seed` is an integer.

**Detail**: within `nH_vqvae.py`, `num_epochs`, `commitment_cost`, `learning_rate`, etc. can all be tuned.

A proper training will yield two files: `models/[model_name]_model.pt` (defaulting to `models/[trainset]_model.pt`) and `codebooks/[model name]_codebook.pkl` (defaulting to `codebooks/[trainset]_codebook.pkl`).

The model file includes the `Encoder()` weights and the codebook vectors, in their latent form.

The codebook file contains an array of these codebook vectors AFTER they have been run through the `Decoder()`, representing the makeup of each cluster.

## `nH_alphabetizer.py`

With `nH_alphabetizer.py`, a dictionary of proteins represented as sequences of nHots can be converted into a dictionary of proteins represented as sequences over the 20 learned VQVAE states.

This conversion can be performed with `python nH_alphabetizer.py <model.pt> <toBeAlphabetized.pkl>`, giving two outputs with the same information.

(1) `[toBeAlphabetized]_alphabeted_with_[model].pkl`
(2) `[toBeAlphabetized]_alphabeted_with_[model].fasta`

(NOTE: BOTH WILL SOON BE REPLACED WITH A SINGLE .npz FILE, AND A .npy MATRIX WILL BE GENERATED AUTOMATICALLY)

Currently, feeding the `.pkl` file to `getMat.py` will yield a BLOSUM matrix in two files (`.npy` and `.out`) along with its associated mutual information, a good heuristic for the performance of the model.
