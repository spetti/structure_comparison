# Neighborhood (cluster) visualizations

`sphereProjection.ipynb` provides a framework to visualize the 20 decoded codebook vectors found in training (see `../vqvae_on_nHots/README.md`).

Ensure that `num_radii_bins`, `num_theta_bins`, and `num_phi_bins` are set to correct values (they are for current 1000-dim nHots).

Save a codebook of shape (20, 1000) as a `[codebook_name].npy` NumPy matrix. Substitute this file in the final codeblock of `sphereProjection.ipynb`.

Images, flatted against phi and against theta will be created and saved to a directory called `./plots/embeddings/`.
