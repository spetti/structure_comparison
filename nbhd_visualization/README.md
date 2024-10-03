# Neighborhood (cluster) visualizations

`sphereProjection.ipynb` provides a framework to visualize the 20 decoded codebook vectors found in training (see `../vqvae_on_nHots/README.md`).

Ensure that `num_radii_bins`, `num_theta_bins`, and `num_phi_bins` are set to correct values (they are for current 1000-dim nHots).

Use the name of the your desired codebook (`<codebook.pkl>`) in the final codeblock.

Images, flatted against phi and against theta will be created and saved to a directory called `./plots/embeddings/`.
