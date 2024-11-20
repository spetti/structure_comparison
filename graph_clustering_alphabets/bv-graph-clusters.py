#!/usr/bin/env python
# coding: utf-8

import math
import csv
import numpy as np
from matplotlib import pyplot as plt
import time
import cProfile
import pickle
import os
import random
from collections import defaultdict

from collections import Counter
import networkx as nx


#get train and validation sets 
def train_val_prots(fileName):
    
    with open(fileName, "rb") as ff:
        dicti_train_val_test = pickle.load(ff)

    trainProts=[]
    valProts=[]
    for prot in dicti_train_val_test.keys():
        if dicti_train_val_test[prot]=='train':
            trainProts.append(prot)
        elif dicti_train_val_test[prot]=='validation':
            valProts.append(prot)
    
    return trainProts, valProts

def bv_subset(trainProts, allProtSphericalBinsWithSecondaryStruct_1Hot, tMtx, n):
    #n is number of bv we want to sample
    
    trainBVList=[]
    for i,prot in enumerate(trainProts):
        sphereInfo=allProtSphericalBinsWithSecondaryStruct_1Hot[prot]

        bv=np.einsum('ik,kj->ij', sphereInfo, tMtx)[:,:-1]
        for row in bv:
            trainBVList.append(row)
        if i%100==0:
            print(f"{i}/{len(trainProts)}")


    random.seed(0)
    bVListSubset=random.sample(trainBVList,n) #randomly pick n (e.g. 20k) to cluster
    
    return bVListSubset


def pairwiseDist(X):
    n = len(X)
    distances = np.zeros((n, n))
    sim = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist=jaccard1D(X[i],X[j])
            distances[i, j] = dist
            sim[i, j] = 1.0 - dist
            distances[j, i] = dist
            sim[j, i] = 1.0 - dist
    
    return distances, sim

def jaccard1D(v1,v2):
    min_elements = np.minimum(v1,v2)
    max_elements = np.maximum(v1,v2)
    
    # Sum over the last dimension to get the final result
    result = np.sum(min_elements)/np.sum(max_elements)
    
    return 1-result



#centroid computation
#returns a list of medioids of clusters, which we will use for centroids
def compute_cluster_medians(distance_matrix, cluster_assignments, referenceList):

    clusters = np.unique(cluster_assignments)
    cluster_medians = {}

    for cluster in clusters:
        indices_in_cluster = np.where(cluster_assignments == cluster)[0]

        intra_cluster_distances = distance_matrix[np.ix_(indices_in_cluster, indices_in_cluster)]

        sum_distances = intra_cluster_distances.sum(axis=1)

        median_index_in_cluster = np.argmin(sum_distances)

        median_index = indices_in_cluster[median_index_in_cluster]
        
        cluster_medians[cluster] = referenceList[median_index]

    return list(cluster_medians.values())

def find_closest_centroid(vector, centroids):
    distances = [jaccard1D(vector, centroid) for centroid in centroids]
    return np.argmin(distances)

def seqFromCentroids(prot,centroids):
    centSeq=[]
    sphereInfo=allProtSphericalBinsWithSecondaryStruct_1Hot[prot]
    bv2=np.einsum('ik,kj->ij', sphereInfo, tMtx)[:,:-1]
    for row in bv2:
        centSeq.append((find_closest_centroid(row,centroids),))
    return centSeq



def create_sparse_graph(G, k): # k-nearest neighbor graph
    sparse_graph = nx.Graph()
    
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        weights = [(neighbor, G[node][neighbor]['weight']) for neighbor in neighbors]
        # Sort by weight and take the top k
        top_neighbors = sorted(weights, key=lambda x: x[1], reverse=True)[:k]
        
        for neighbor, weight in top_neighbors:
            sparse_graph.add_edge(node, neighbor, weight=weight)
    
    return sparse_graph

def graph_clusters(G,res):
    louvain_clust = nx.community.louvain_communities(G, weight='weight', resolution=res, seed=123)
    n_c = len(louvain_clust)
    n_nodes = nx.number_of_nodes(G)
    louvain_list = np.zeros(n_nodes)
    louvain_list = louvain_list.astype(int)

    for i in range(len(louvain_clust)):
        for j in list(louvain_clust[i]):
            louvain_list[j] = i
    return louvain_list, n_c



#convert to 1-Hot
def getIndex(l,value):
    try: 
        return l.index(value)
    except: 
        return 'e'

def getOneHotCent(baseList,oneHotVals):
    leng=len(baseList)
    n=len(oneHotVals)
    oneHotMtx=np.zeros((leng,n))
    for i in range(leng):
        ind=getIndex(oneHotVals,baseList[i])
        oneHotMtx[i][ind]=1
    return oneHotMtx


def main():

    trainProts, valProts = train_val_prots("/cluster/tufts/pettilab/shared/structure_comparison_data_old/train_test_val/dicti_train_val_test.pkl")
    tMtx=np.loadtxt('/cluster/tufts/pettilab/shared/structure_comparison_data_old/transitionMtx.txt', delimiter=',')

    with open('/cluster/tufts/pettilab/shared/structure_comparison_data_old/allProtSphericalBinsWithSecondaryStruct_1Hot.pkl','rb') as f:
        allProtSphericalBinsWithSecondaryStruct_1Hot=pickle.load(f)



    #random sample of bV's
    n_bv = 20000
    bVListSubset = bv_subset(trainProts, allProtSphericalBinsWithSecondaryStruct_1Hot, tMtx, n_bv)


    #save these blurry vectors
    bVSubsetDict = {str(k) : v for k, v in enumerate(bVListSubset)}
    np.savez(f'/cluster/tufts/pettilab/shared/structure_comparison_data/graph_cluster_data/bv_subset_{n_bv}.npz',**bVSubsetDict)



    #get pairwise distance, similarity 
    start = time.time()
    dMtx, SimMtx = pairwiseDist(bVListSubset)
    print(f"{time.time()-start}")


    # Adjacecny matrix with chosen threshold parameter
    A = SimMtx.copy()
    A = np.matrix(A)
    A[A < 0.5] = 0.0 # discard low weight links

    #construct a graph using this adjacency
    G=nx.from_numpy_array(A)

    # k-nearset neighbor graph
    kn = 20
    # create a much sparser graph by retaining only kn highest weight neighbors for each node
    G = create_sparse_graph(G, kn) 


    clusterLists = []
    n_clusters = []

    resParameters=[1.3, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 23.5]

    for i in range(len(resParameters)):
        louvain_list, n_c = graph_clusters(G, resParameters[i])
        clusterLists.append(louvain_list)
        n_clusters.append(n_c)
        np.save(f'/cluster/tufts/pettilab/shared/structure_comparison_data/graph_cluster_data/cluster_labels_{n_clusters[i]}.npy',louvain_list)


    for i in range(len(clusterLists)):
        counts=list(Counter(clusterLists[i]).values())
        print(len(np.array(counts)))
        print(np.array(counts))


    centroidsList=[]
    for i in range(len(clusterLists)):
        centroidsList.append(compute_cluster_medians(dMtx,clusterLists[i],bVListSubset))
        
        
    allProts = list(allProtSphericalBinsWithSecondaryStruct_1Hot.keys())

    tickSepList = []
    centLettersList = []
    for i in range(len(centroidsList)):    
        tickSep1=[j for j in range(len(centroidsList[i]))]
        centLetters1=[]
        for j in range(len(centroidsList[i])):
            centLetters1.append((j,))
        centLettersList.append(centLetters1)
        tickSepList.append(tickSep1)


    # Regenerating Dictionaries but with All proteins:
    dictsProtstoCentroidsAll=[]
    for i in range(len(centroidsList)):
        protDict={}
        for j,prot in enumerate(allProts):
            if j%100==0:
                print(f"round {i}, {j}/{len(allProts)}")
            protDict[prot]=seqFromCentroids(prot,centroidsList[i])
        dictsProtstoCentroidsAll.append(protDict)


    dictsOneHotAll=[]
    for i in range(len(centroidsList)):
        dictOneHot={}
        refDict=dictsProtstoCentroidsAll[i]
        centLetters = centLettersList[i]
        for prot in allProts:
            dictOneHot[prot]=getOneHotCent(refDict[prot],centLetters)
        dictsOneHotAll.append(dictOneHot)
        print(i)
        

    for i in range(len(centroidsList)):
        protDict1Hot=dictsOneHotAll[i]
        protDict1Hot={key: value for key, value in protDict1Hot.items() if key not in ['d1o7d.2', 'd1o7d.3']} # delete the two proteins oh_d1['d1o7d.2'],oh_d1['d1o7d.3']
        np.savez(f'/cluster/tufts/pettilab/shared/structure_comparison_data/diff_size_alphabets_P/graph_clusters_{n_clusters[i]}.npz',**protDict1Hot)


if __name__ == "__main__":
    main()