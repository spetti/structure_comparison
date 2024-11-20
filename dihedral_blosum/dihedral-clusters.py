#!/usr/bin/env python
# coding: utf-8

import copy
import random
import math
import csv
import numpy as np
from matplotlib import pyplot as plt
import warnings
import scipy
import scipy.stats
import scipy.signal
import time
import cProfile
import json
from Bio.PDB import *
import os
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from Bio import BiopythonDeprecationWarning
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq3
from Bio.PDB.Polypeptide import *
from scipy.cluster import hierarchy
import pickle


def getphipsi(filesTruncated): # for a number of proteins so we can get high freq bins
    
    p = PDBParser()

    phis=[]
    psis=[]
    glyphi=[]
    glypsi=[]
    prophi=[]
    propsi=[]
    for i in range(len(filesTruncated)):
        fileX=path + '/' + filesTruncated[i]
        structure = p.get_structure("my_protein", fileX)
        model = structure[0]  # assuming we're interested in the first model
        ppb = PPBuilder()    
        for chain in model:
            polypeptides = ppb.build_peptides(chain)
            for poly in polypeptides:
                phi_psi = poly.get_phi_psi_list()
                for res_index, residue in enumerate(poly):
                    phi, psi = phi_psi[res_index]
                    if phi and psi:  # if both angles are not None
                        #print(f"Residue: {residue.resname}, Phi: {phi:.2f}, Psi: {psi:.2f}")
                        phis.append(phi)
                        psis.append(psi)
                        if residue.get_resname()=='GLY':
                            glyphi.append(phi)
                            glypsi.append(psi)
                        if residue.get_resname()=='PRO':
                            prophi.append(phi)
                            propsi.append(psi)
                            
    return phis, psis, glyphi, glypsi, prophi, propsi


def angleToCategory(theta, n):
    c = int(np.floor((theta+np.pi)/(2*np.pi/(n))))
    if c > 29:
        c = c - 30
    if c < 0:
        c = c + 30
    return c



def highFreqBins(filesTruncated, numberOfBins):
    
    
    phis, psis, glyphi, glypsi, prophi, propsi = getphipsi(filesTruncated)
    
    
    #count bins for each element in phipsi
    phiPsiCounts={}
    totalCount=0
    for i in range(numberOfBins):
        for j in range(numberOfBins):
            phiPsiCounts[(i,j)]=0

    for i in range(len(phis)):
        phiBin=angleToCategory(phis[i],numberOfBins)
        psiBin=angleToCategory(psis[i],numberOfBins)
        phiPsiCounts[(phiBin,psiBin)]+=1
        totalCount+=1

    for i in range(numberOfBins):
        for j in range(numberOfBins):
            phiPsiCounts[(i,j)]=phiPsiCounts[(i,j)]/totalCount 

    ################################


    #count bins for each element in phipsi PRO
    prophiPsiCounts={}
    prototalCount=0
    for i in range(numberOfBins):
        for j in range(numberOfBins):
            prophiPsiCounts[(i,j)]=0

    for i in range(len(prophi)):
        prophiBin=angleToCategory(prophi[i],numberOfBins)
        propsiBin=angleToCategory(propsi[i],numberOfBins)
        prophiPsiCounts[(prophiBin,propsiBin)]+=1
        prototalCount+=1

    for i in range(numberOfBins):
        for j in range(numberOfBins):
            prophiPsiCounts[(i,j)]=prophiPsiCounts[(i,j)]/prototalCount 


    #count bins for each element in phipsi GLY
    glyphiPsiCounts={}
    glytotalCount=0
    for i in range(numberOfBins):
        for j in range(numberOfBins):
            glyphiPsiCounts[(i,j)]=0

    for i in range(len(glyphi)):
        glyphiBin=angleToCategory(glyphi[i],numberOfBins)
        glypsiBin=angleToCategory(glypsi[i],numberOfBins)
        glyphiPsiCounts[(glyphiBin,glypsiBin)]+=1
        glytotalCount+=1

    for i in range(numberOfBins):
        for j in range(numberOfBins):
            glyphiPsiCounts[(i,j)]=glyphiPsiCounts[(i,j)]/glytotalCount 



    ################################

    # count top bins to be used for sorting
    # same threshold used for GLY and PRO separately

    accountedForPercentage=0
    highFrequencyPhiPsis=[]
    for i in range(numberOfBins):
        for j in range(numberOfBins):
            if phiPsiCounts[(i,j)]>1/900:
                accountedForPercentage+=phiPsiCounts[(i,j)]
                highFrequencyPhiPsis.append((i,j))

            elif prophiPsiCounts[(i,j)]>1/900:
                accountedForPercentage+=phiPsiCounts[(i,j)]
                highFrequencyPhiPsis.append((i,j))

            elif glyphiPsiCounts[(i,j)]>1/900:
                accountedForPercentage+=phiPsiCounts[(i,j)]
                highFrequencyPhiPsis.append((i,j))

    print('number of bins ' + str(len(highFrequencyPhiPsis)))
    print('the number of bins account for ' + str(accountedForPercentage))
    
    return highFrequencyPhiPsis


def calc_phi_psi(prev_residue, residue, next_residue):
    try:
        C_prev = prev_residue["C"].get_vector()
        
        N = residue["N"].get_vector()
        CA = residue["CA"].get_vector()
        C = residue["C"].get_vector()
        
        N_next = next_residue["N"].get_vector()
        
        phi = calc_dihedral(C_prev, N, CA, C)
        psi = calc_dihedral(N, CA, C, N_next)
        
        return phi, psi 
    except KeyError:
        # Not all residues have all atoms, catch exceptions here
        return '-','-'


def backbone_from_pdb(fileName,highVals):
    angles=[]
    p = PDBParser()
    structure = p.get_structure("my_protein", fileName)
    for model in structure:
        for chain in model:
            residues =  list(chain)
            angles.append(('-','-'))
            for i in range(1, len(residues)-1):
                phi, psi = calc_phi_psi(residues[i - 1], residues[i], residues[i + 1])
                phiCategory=angleToCategory(phi, numberOfBins)
                psiCategory=angleToCategory(psi, numberOfBins)
                if (phiCategory,psiCategory) in highVals:
                    phi=phiCategory
                    psi=psiCategory
                    angles.append((phi,psi))
                else: #assign to one bin
                    angles.append((-1,-1))
            angles.append(('-','-'))
    return angles


#########################
#These functions help process data from files:

#reads a file in same format that dali uses and returns dict: dic[pairs]=(firstCigarStarts, secondCigarStarts, cigars)
def parseFile(fileName):
    fileInfo={}
    with open (fileName,'r') as file:
        reader=csv.reader(file,delimiter='\t')
        for row in reader:
            fileInfo[(row[0],row[1])]=(int(row[2]),int(row[3]),row[4])
    return fileInfo

#parseFile but using TM align
#returns dict: dic[pair]=TM-alignment
def parseFile_TM(fileName):
    fileInfo={}
    with open (fileName,'r') as file:
        reader=csv.reader(file,delimiter=',')
        next(reader)
        for i,row in enumerate(reader):
            try:
                algt_str=row[4]
                algt_str=algt_str.strip('[]')
                algt_str=algt_str.split()
                #algt=list(map(int, algt_str))
                algt=[int(s) for s in algt_str]
                fileInfo[(row[1],row[2])]=algt
            except ValueError:
                pass
    return fileInfo
    

######################
#These functions are all helpers for acquiring and analyzing alignment data

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

#returns the amino acid sequence of a protein with file path "pdb_file"
def sequence_from_pdb(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    seq = []
    for model in structure:
        for chain in model:
            for residue in chain:
                try: 
                    seq.append(three_to_one(residue.get_resname()))
                except KeyError:
                    seq.append('X')
    return seq

#Parses a cigar string and returns its infromation with a list of tuples
def parse_cigar(cigar):
    number = ''
    result = []
    for char in cigar:
        if char.isdigit():
            number += char
        else:
            result.append((char, int(number)))
            number = ''
    return result

#Creates a list of two equal length strings that align corresponding amino acids of two proteins
#based on cigar
def cig_visualizer(prot1,prot2,start1,start2,cig):
    cig_relevant=parse_cigar(cig)
    index1=start1
    index2=start2
    list1=[]
    list2=[]
    for i in cig_relevant:
        letter=i[0]
        num=i[1]
        if letter=='M':
            list1.extend(prot1[index1-1:index1+num-1])
            list2.extend(prot2[index2-1:index2+num-1])
            index1+=num
            index2+=num
        elif letter=='D':
            for j in range(num):
                list1.extend('-')
            list2.extend(prot2[index2-1:index2+num-1])
            index2+=num
        elif letter=='I':
            for k in range(num):
                list2.extend('-')
            list1.extend(prot1[index1-1:index1+num-1])
            index1+=num
    return (list1,list2)

#Creates a list of two equal length strings that align corresponding amino acids of two proteins
#based on cigar
def tmAlign_visualizer(tmArr):
    list1=[]
    list2=[]
    ind1=0
    ind2=0
    for indexProtA,i in enumerate(tmArr):
        if i==-1:
            list2.append('-')
            list1.append(indexProtA)
        elif len(list1)==0:
            list1.append(indexProtA)
            list2.append(i)
        elif len(list1)>=0:
            if list2[-1]!='-' and list2[-1]+1!=i:
                for b in range(list2[-1]+1,i):
                    list2.append(b)
                    list1.append('-')
                list2.append(i)
                list1.append(indexProtA)
            else:
                list1.append(indexProtA)
                list2.append(i)
    return (list1,list2)
                
#like tmAlign_visualizer but specific to the sequence IDs of amino acids
def alignPairTM(tmArr,seq1,seq2):
    list1,list2=tmAlign_visualizer(tmArr)
    res1=[]
    res2=[]
    for a in range(len(list1)):
        if list1[a]!='-':
            res1.append(seq1[list1[a]])
        elif list1[a]=='-':
            res1.append('-')
        if list2[a]!='-':
            res2.append(seq2[list2[a]])
        elif list2[a]=='-':
            res2.append('-')
    return (res1,res2)
            
    

#Determine the percentage match (#of same amino acids in same position/length of amino acid) of two aligned sequences
def percentSimilarity(protStr1,protStr2,labelsArr):
    count=0
    for i in range(len(protStr1)):
        if protStr1[i]==protStr2[i] and protStr1[i] in labelsArr:
            count+=1
    return count/len(protStr1)

def preprocessFile(file,parserDict):
    fileInfo=parseFile(file)
    dataList=[]
    for key in fileInfo:
        aa1=parserDict[key[0]]
        aa2=parserDict[key[1]]
        cigInfo=fileInfo[key]
        firstCigarStart=cigInfo[0]
        secondCigarStart=cigInfo[1]
        cigar=cigInfo[2]
        dataList.append(cig_visualizer(aa1,aa2,firstCigarStart,secondCigarStart,cigar))
    return dataList

def preprocessFileTM(file,parserDict):
    fileInfo=parseFile_TM(file)
    dataList=[]  
    for key in fileInfo:
        seq1=parserDict[key[0]]
        seq2=parserDict[key[1]]
        tmArr=fileInfo[key]
        dataList.append(alignPairTM(tmArr,seq1,seq2))
    return dataList
        

#make a BLOSUM dictionary
def makeBLOSUM(data, labelsArr):
    singleCounts={}
    pairCounts={}
    blosum={}
    
    for i in labelsArr:
        singleCounts[tuple(i)]=0
        for j in labelsArr:
            index=frozenset([tuple(i),tuple(j)])
            pairCounts[index]=0
            blosum[index]=0
    #print(pairCounts.keys())
    
    totalSingleCount=0
    totalPairCount=0
    
    goodSet=set(labelsArr)
    for row in data:
        seq1=row[0]
        seq2=row[1]
        for i in range(len(seq1)):
            element1=(seq1[i],)
            element2=(seq2[i],)
#             print(element1)
#             print(element2)
#             print(goodSet)
            if element1 in goodSet and element2 in goodSet:
                totalSingleCount+=2
                totalPairCount+=1
                pairCounts[frozenset([element1,element2])]+=1
                singleCounts[element1]+=1
                singleCounts[element2]+=1

    for key in singleCounts.keys():
        singleCounts[key]=singleCounts[key]/totalSingleCount
    for key in pairCounts.keys():
        pairCounts[key]=pairCounts[key]/totalPairCount
        
    for key in blosum.keys():
        multTwo=True
        if len(key)==2:
            i1,i2=key
            i1=tuple(i1)
            i2=tuple(i2)
        else:
            (i1,)=key
            i2=i1
            multTwo=False
        try:
            blosum[key]=round(2.0*math.log(pairCounts[key]/(singleCounts[i1]*singleCounts[i2]*(multTwo+1)))/math.log(2))
        except ValueError:
            blosum[key]=-10

        except ZeroDivisionError:
            blosum[key]=-10
#             print(i1)
#             print(i2)
    return blosum

#turn the BLOSUM dictionary into a visualizable array
def BLOSUMDictToArray(blosum,labelsArr):
    blosumArr=np.zeros((len(labelsArr),len(labelsArr)))
    for x,i in enumerate(labelsArr):
        for y,j in enumerate(labelsArr):
            curPair=frozenset([i,j])
            blosumArr[x,y]=blosum[curPair]
    return blosumArr

#plot BLOSUM array, including x/y labels
def plotBLOSUM(blosumArr,tickSep,labelsArr,title):
    plt.imshow(blosumArr,vmin=-10,vmax=10,cmap='bwr')
    plt.xticks(tickSep, labels=labelsArr)
    plt.yticks(tickSep, labels=labelsArr)
    plt.title(title)
    plt.colorbar()
    plt.show()

#plot BLOSUM array, excluding x/y labels
def plotBLOSUMMinimal(blosumArr,title):
    plt.imshow(blosumArr,vmin=-10,vmax=10,cmap='bwr')
    plt.title(title)
    plt.colorbar()
    plt.show()
    
#plot blosum with superimposed grid; mainly useful for rotamer blosum
def plotBLOSUMRot(blosumArr, tickSep,labelsArr,dictLabelsArr,title):
    fig, ax = plt.subplots(figsize=(7.5,7.5))
    ax.imshow(blosumArr,vmin=-10,vmax=10,cmap='bwr')

    prev='A'
    for i in range(len(dictLabelsArr)):
        relevantLetter=dictLabelsArr[i][0]
        if prev!=relevantLetter:
            plt.axvline(i-1,color='k',linewidth=1)
            plt.axhline(i-1,color='k',linewidth=1)
        prev=relevantLetter

    ax.set_yticks(tickSep, labels=labelsArr,fontsize=12)
    ax.set_xticks(tickSep, labels=labelsArr,fontsize=12)
    fig.colorbar(ax.images[0])

    plt.title(title,fontsize=15)
    plt.show()

#determine the score of an AA sequence/analog of AA sequence using BLOSUM 
def scoreBLOSUMPair(blosumDict,seq1,seq2,labelsArr,gapPenFirst,gapPenRest):
    gapPrev=False
    totalScore=0
    discounted=0
    
    for i in range(len(seq1)):
        firstElement=tuple(seq1[i])
        secondElement=tuple(seq2[i])
        if '-' in firstElement or '-' in secondElement:
            if gapPrev==False:
                gapPrev=True
                totalScore+=gapPenFirst
            else:
                totalScore+=gapPenRest
        elif 'X' in firstElement or 'X' in secondElement:
            discounted+=1
            gapPrev=False
        else:
            gapPrev=False
            index=frozenset([firstElement,secondElement])
            totalScore+=blosumDict[index]
    return totalScore/(len(seq1)-discounted)

#determine the score of an entire paradigm using BLOSUM, compares to TM score
def scoreBLOSUM(TMdict,blosumDict,file,parserDict,labelsArr,gapPenFirst,gapPenRest):
    tmList=[]
    blosumList=[]
    fileInfo=parseFile(file)
    for pair in fileInfo.keys():
        indexPair=frozenset([pair[0],pair[1]])
        if indexPair in TMdict.keys():
            tmList.append(TMdict[indexPair])
            cigarInfo=fileInfo[pair]
            seq1,seq2=cig_visualizer(parserDict[pair[0]],parserDict[pair[1]],cigarInfo[0],cigarInfo[1],cigarInfo[2])
            blosumList.append(scoreBLOSUMPair(blosumDict,seq1,seq2,labelsArr,gapPenFirst,gapPenRest))
    return tmList, blosumList

#plot blosum scoring correlation given tmlist and affiliated blosum scores
def plotBLOSUMScore(tmList,blosumScoreList,title):
    tmList=np.array(tmList,dtype=float)    
    blosumScoreList=np.array(blosumScoreList,dtype=float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(tmList, blosumScoreList)
    line=slope*tmList+intercept

    plt.scatter(tmList,blosumScoreList,color='blue',s=3)
    plt.plot(tmList, line, color='red', label=f'y={slope:.2f}x+{intercept:.2f}, R2={r_value**2:.2f}')
    plt.legend()
    plt.title(title)
    plt.xlabel('TM Score')
    plt.ylabel('BLOSUM Score')
    plt.xlim(0.59,1)
    plt.ylim(-1,3)
    plt.show()
    
################################################

## Exactly as BLOSUM except it calculates the probabilities
def makePIJ(data, labelsArr):
    singleCounts={}
    pairCounts={}
    blosum={}
    
    for i in labelsArr:
        singleCounts[tuple(i)]=0
        for j in labelsArr:
            index=frozenset([tuple(i),tuple(j)])
            pairCounts[index]=0
            blosum[index]=0
    
    totalSingleCount=0
    totalPairCount=0
    
    goodSet=set(labelsArr)
    for row in data:
        seq1=row[0]
        seq2=row[1]
        for i in range(len(seq1)):
            element1=(seq1[i],)
            element2=(seq2[i],)

            if element1 in goodSet and element2 in goodSet:
                totalSingleCount+=2
                totalPairCount+=1
                pairCounts[frozenset([element1,element2])]+=1
                singleCounts[element1]+=1
                singleCounts[element2]+=1

    for key in singleCounts.keys():
        singleCounts[key]=singleCounts[key]/totalSingleCount
    for key in pairCounts.keys():
        pairCounts[key]=pairCounts[key]/totalPairCount
        
    for key in blosum.keys():
        multTwo=True
        if len(key)==2:
            i1,i2=key
            i1=tuple(i1)
            i2=tuple(i2)
        else:
            (i1,)=key
            i2=i1
            multTwo=False
        try:
            blosum[key]=pairCounts[key]/(multTwo+1)
        except ValueError:
            blosum[key]=-10

        except ZeroDivisionError:
            blosum[key]=-10
#             print(i1)
#             print(i2)
    return blosum


def get_indices(DihedralVals):
    matindex = {}
    for i, j in enumerate(DihedralVals):
        matindex[i]=j

    matindex[len(matindex)]=('-','-')
    #matindex['-']=('-','-')
    matindex[-1]=(-1,-1) #added this extra bin (not used when finding neighbors)


    inv_matindex = {v: k for k, v in matindex.items()}
    
    return matindex, inv_matindex


def get_matrixA(DihedralVals,allDaliBackboneDihedralAngles,trainingfile):
    
    matindex, inv_matindex = get_indices(DihedralVals)
    
    #Dihedral bins
    dihedral_bins={}
    for x in allDaliBackboneDihedralAngles.keys():
        dihedral_bins[x] = [inv_matindex[j] for j in allDaliBackboneDihedralAngles[x]]


    #bin Tuples
    binTuples=[(i,) for i in range(252)]

    blosumDiBins=makePIJ(preprocessFileTM(trainingfile,dihedral_bins),binTuples)
    A=BLOSUMDictToArray(blosumDiBins,binTuples)

    A = np.matrix(np.array(A))
    
    return A


def get_neighbors(DihedralVals):
        
    matindex = {}
    for i, j in enumerate(DihedralVals):
        matindex[i]=j


    inv_matindex = {v: k for k, v in matindex.items()}

    #torus
    neighbors = {n:set() for n in matindex.keys()}
    for i in matindex.keys():    
        for j in matindex.keys():
            if (len(matindex[i])==2 and len(matindex[j])==2 and abs(matindex[i][0]-matindex[j][0])+abs(matindex[i][1]-matindex[j][1])==1):
                neighbors[i].add(j)
            elif (len(matindex[i])==2 and len(matindex[j])==2 and matindex[i][0]-matindex[j][0]==0 and abs(matindex[i][1]-matindex[j][1])==29):
                neighbors[i].add(j)
            elif (len(matindex[i])==2 and len(matindex[j])==2 and matindex[i][1]-matindex[j][1]==0 and abs(matindex[i][0]-matindex[j][0])==29):
                neighbors[i].add(j)
    neighbors[251] = set()
    
    return matindex, inv_matindex, neighbors



def calc_ent1(A):
    pi = np.array(A.sum(1)).flatten()
    pj = np.array(A.sum(0)).flatten()
    s=0.0
    for i in range(len(A)):
        for j in range(len(A)):
            
            if A[i,j]>0:
                s=s+A[i,j]*np.log(A[i,j]/(pi[i]*pj[j]))
    return s

def findPair(A_before,neighbors):
    tomerge1=None
    tomerge2=None
    maxH=0.0
    for i in range(len(A_before)):
        for j in neighbors[i]:
            if (i < j and np.matrix.sum(A_before[i])>0 and np.matrix.sum(A_before[j])>0):
                
                A_after = A_before.copy()
                A_after[i]=A_before[i]+A_before[j]
                A_after[:,i]=A_before[:,i]
                A_after[:,i]=A_before[:,i] + A_before[:,j]

                A_after[j]=0.0
                A_after[:,j]=0.0
                
                H = calc_ent1(A_after)
                
                if(H > maxH):
                    maxH=H
                    tomerge1=i
                    tomerge2=j
    return tomerge1, tomerge2, maxH



def findClusters(A, neighbors):
    
    clusters = list(neighbors.keys())
    cluster_dict = dict(zip(neighbors.keys(),clusters)) #initialize

    cluster_dict_t = {} # clusters at every t

    new_clusters = clusters 

    A_before = A.copy()

    new_neighbors=neighbors.copy()


    n=0
    #n_list=[]
    bin1=[]
    bin2=[]
    #H_list = []

    condition=True

    while condition:

        i,j,H = findPair(A_before,new_neighbors)
        #print(i,j,H)
        if(i is not None):

            A_after = A_before.copy()
            A_after[i]=A_before[i]+A_before[j]

            A_after[:,i]=A_before[:,i] + A_before[:,j]

            A_after[j]=0.0
            A_after[:,j]=0.0

            new_neighbors[i]=new_neighbors[i].union(new_neighbors[j])
            for z in new_neighbors[j]:
                new_neighbors[z]=new_neighbors[z].union({i})


            A_before = A_after.copy()


            clusters = [clusters[i] if x==clusters[j] else x for x in clusters]

            temp = {l: m for m, l in enumerate(set(clusters))} 
            new_clusters = [temp[l] for l in clusters] 
            cluster_dict = dict(zip(neighbors.keys(),new_clusters))

            cluster_dict_t[n]=cluster_dict # dictionry of clusters at step t

            clusters=new_clusters

            bin1.append(i)
            bin2.append(j)

            #n_list.append(n)
            #H_list.append(H)
            n=n+1
        else:
            condition=False
            
    return bin1, bin2, cluster_dict_t


def makeRamaPlot(t, cluster_dict_t, bin1, bin2):
    numberOfBins=30
    if (t <= len(cluster_dict_t)):
        
    
        plt.figure(figsize=(10, 10))

        plt.title("Ramachandran Plot (nc = {})".format(len(DihedralVals)-t),fontsize=20)
        plt.xlabel('phi_k [rad]',fontsize=15)
        plt.ylabel('psi_k [rad]',fontsize=15)

        # # Set the x and y axis scales to cover the full range of possible phi/psi values
        plt.xlim(-np.pi, np.pi)
        plt.ylim(-np.pi, np.pi)

        # # Customize the tick marks and labels on the x and y axis
        #plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['-π', '-π/2', '0', 'π/2', 'π'])
        #plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['-π', '-π/2', '0', 'π/2', 'π'])
        x=np.linspace(-np.pi,np.pi,numberOfBins+1)
        y=x
        plt.xticks(x)
        plt.yticks(y)
        xtick_labels = ['' for _ in x] # first create a list of empty strings
        ytick_labels = ['' for _ in y] # for y-axis
        xtick_labels[::5] = [f'{tick:.2f}' for tick in x[::5]] # now every fifth element is the respective tick value
        ytick_labels[::5] = [f'{tick:.2f}' for tick in y[::5]] # for y-axis
        plt.gca().set_xticklabels(xtick_labels,fontsize=10)
        plt.gca().set_yticklabels(ytick_labels,fontsize=10)

        for i in range(0,t):

            x1,y1=((2*np.pi)/60)-np.pi+matindex[bin1[i]][0]*(2*np.pi)/30, ((2*np.pi)/60)-np.pi+matindex[bin1[i]][1]*(2*np.pi)/30
            x2,y2=((2*np.pi)/60)-np.pi+matindex[bin2[i]][0]*(2*np.pi)/30, ((2*np.pi)/60)-np.pi+matindex[bin2[i]][1]*(2*np.pi)/30

            plt.scatter(x1,y1,marker='s',s=240,color="gray")
            plt.scatter(x2,y2,marker='s',s=240,color="gray")


        for i in range(len(DihedralVals)):
            j = cluster_dict_t[t-1][inv_matindex[DihedralVals[i]]]
            x,y=((2*np.pi)/60)-np.pi+DihedralVals[i][0]*(2*np.pi)/30, ((2*np.pi)/60)-np.pi+DihedralVals[i][1]*(2*np.pi)/30

            plt.text(x,y,cluster_dict_t[t-1][inv_matindex[DihedralVals[i]]],fontsize=6)

            plt.scatter(x,y,marker='s',s=240,color=plt.cm.tab20(j))


        plt.grid(True)
    
    else:
        print('select a smaller number of steps (1st argument).')

    plt.savefig('/cluster/tufts/pettilab/shared/structure_comparison_data/dihedral/RamaPlot-{}-clusters-{}-bins-MI-contiguous.png'.format(len(DihedralVals)-t,len(DihedralVals)),format='png',bbox_inches='tight')


def get_dihedral_letters(DihedralVals, cluster_dict_t, nc):
    
    new_dict = {}

    matindex = {}
    for i, j in enumerate(DihedralVals):
        matindex[i]=j

    matindex[len(matindex)]=('-','-') #end points

    matindex[-1]=(-1,-1)


    inv_matindex = {v: k for k, v in matindex.items()}



    for k, v in cluster_dict_t[len(DihedralVals) - nc - 1].items():
        #print(k, v)
        new_dict[matindex[k]] = v
    
    new_dict[(-1,-1)]=19# # everything outside the high freq. bins

    return new_dict


# Convert to 1-Hot
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
    #path to pdb files # can't locate
    path='/cluster/tufts/pettilab/shared/structure_comparison_data/protein_data/ProteinDB/pdb/'

    files=os.listdir(path)
    filesTruncated=files # or use a smaller set


    numberOfBins=30
    highFrequencyPhiPsis = highFreqBins(filesTruncated, numberOfBins)
    DihedralVals = highFrequencyPhiPsis


    trainingfile = '/cluster/tufts/pettilab/shared/structure_comparison_data/protein_data/pairs_training.csv'

    allDaliProts=files

    allDaliBackboneDihedralAngles={}
    for prot in allDaliProts:
        fileName=path+prot
        #print(prot)
        allDaliBackboneDihedralAngles[prot]=backbone_from_pdb(fileName,highFrequencyPhiPsis)


    # write to a file
    #with open('allDaliBackboneDihedralAngles_30Bins','w') as f:
    #    json.dump(allDaliBackboneDihedralAngles,f)

    matindex, inv_matindex, neighbors = get_neighbors(DihedralVals)

    A = get_matrixA(DihedralVals,allDaliBackboneDihedralAngles,trainingfile)


    bin1, bin2, cluster_dict_t = findClusters(A, neighbors)


    makeRamaPlot(233, cluster_dict_t, bin1, bin2)


    new_dict = get_dihedral_letters(DihedralVals, cluster_dict_t, 18)

    with open('/cluster/tufts/pettilab/shared/structure_comparison_data/dihedral/new_dict.txt', 'wb') as handle:
        pickle.dump(new_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cluster_dict={}
    for x in allDaliBackboneDihedralAngles.keys():
        cluster_dict[x] = [new_dict[j] for j in allDaliBackboneDihedralAngles[x]]

    # convert to 1-Hot
    centLetters = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    dictOneHot={}
    for prot in allDaliProts:
        dictOneHot[prot]=getOneHotCent(cluster_dict[prot],centLetters)


    #remove the two protein and save npz
    dictOneHot = {key: value for key, value in dictOneHot.items() if key not in ['d1o7d.2', 'd1o7d.3']}
    np.savez(f'/cluster/tufts/pettilab/shared/structure_comparison_data/alphabets/dihedral.npz',**dictOneHot)


if __name__ == "__main__":
    main()
