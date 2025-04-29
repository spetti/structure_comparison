# Structure Comparison

This repository contains the source code for "An interpretable alphabet for local protein structure search based on amino acid neighborhoods."

We suggest getting started with our example folder, `3Dn_example`. This folder supplies the script `3Dn_example.ipynb`, which demonstrates the process of generating a 3Dn alignment from two PDB files.

This repository contains the following folders:
- `3Dn_example`, which contains the example script
- `benchmarking`, which contains infrastructure to benchmark a structure character alphabet and combine it with other alphabets (e.g. 3Di, 3Dn, AA/BLOSUM)
- `blosum_construction`, which details the construction of a 3Dn BLOSUM-like matrix using training alignments
- `blurry_vector_construction`, which analyzes local structure information of proteins to construct their blurry neighborhoods
- `compare_and_interpret_alphabets`, which shows comparison of 3Dn and 3Di alphabets
- `data_processing`, which details the handling and type conversion of data necessary for the training and testing processes
- `dihedral_blosum`, which shows the construction and training of a dihedral BLOSUM
- `graph_clustering_alphabets`, which shows the graph clustering processes used to cluster dihedral and blurry neighborhood alphabets
- `nbhd_visualization`, which allows for visualizing blurry neighborhoods
- `vqvae_on_nhots`, which shows the process of training blurry neighborhoods from n-hot vectors using a VQ-VAE

If you use this code, you are welcome to cite us.

```
@article {zerefa2025,
	author = {Zerefa, Saba and Cool, Jesse and Singh, Pramesh and Petti, Samantha},
	title = {An interpretable alphabet for local protein structure search based on amino acid neighborhoods},
	year = {2025},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/04/24/2025.04.21.649886},
	journal = {bioRxiv}
}
```
