# Structure Comparison

This repository contains the source code for "An interpretable alphabet for local protein structure search based on amino acid neighborhoods."

This repository contains the following folders:
- `benchmarking,` which contains infrastructure to benchmark a structure character alphabet and combine it with other alphabets (e.g. 3Di, 3Dn, AA/BLOSUM)
- `blosum_construction,` which details the construction of a 3Dn BLOSUM-like matrix using training alignments
- `blurry_vector_construction,` which analyzes local structure information of proteins to construct their blurry neighborhoods
- `compare_and_interpret_alphabets,` which shows comparison of 3Dn and 3Di alphabets
- `data_processing,` which details the handling and type conversion of data necessary for the training and testing processes
- `dihedral_blosum,` which shows the construction and training of a dihedral BLOSUM
- `graph_clustering_alphabets,` which shows the graph clustering processes used to cluster dihedral and blurry neighborhood alphabets
- `nbhd_visualization,` which allows for visualizing blurry neighborhoods
- `vqvae_on_nhots,` which shows the process of training blurry neighborhoods from n-hot vectors using a VQ-VAE

If you use this code, you are welcome to cite us.
