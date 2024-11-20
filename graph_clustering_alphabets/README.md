## Code to construct alphabets from graph clustering methods

`bv-graph-clusters.py` constructs alphabet by using graph clustering the blurry vectors.

Input data:

`allProtSphericalBinsWithSecondaryStruct_1Hot.pkl`, 1-hot pickle file with spherical bins for each protein. `/cluster/tufts/pettilab/shared/structure_comparison_data_old/`

`dicti_train_val_test.pkl`, pickle file containing training and validation protein sets.

`transitionMtx.txt`, Transition matrix to construct blurry vectors. `/cluster/tufts/pettilab/shared/structure_comparison_data_old/`

Output data:

`graph_clusters_n.npz`, 1-hot `.npz` file where n is the alphabet size. `/cluster/tufts/pettilab/shared/structure_comparison_data/diff_size_alphabets_P/`

`bv_subset_n_bv.npz`, subset of blurry vectors used for clustering. `n_bv` is the number of blurry vectors. `/cluster/tufts/pettilab/shared/structure_comparison_data/graph_cluster_data/`

`cluster_labels_n.npy`, `.npy` file containing cluster labels of each blurry vector in the sample. n is the alphabet size (or the number of clusters). `/cluster/tufts/pettilab/shared/structure_comparison_data/graph_cluster_data/`
