# getMat.py
# DEPRECIATED!!!! SLOW AND OVERDONE

import copy
import os
import math
import csv
import numpy as np
from matplotlib import pyplot as plt
import warnings
import scipy
import scipy.stats as stats
import scipy.signal
import time
import cProfile
import pickle
import json
from Bio.PDB import *
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from Bio import BiopythonDeprecationWarning
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq3
from Bio.PDB.Polypeptide import *
import seaborn as sns

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
def plotBLOSUM(blosumArr,tickSep,labelsArr,title, mutual_info=-1, save_path='.'):
    plt.imshow(blosumArr,vmin=-30,vmax=30,cmap='bwr')
    plt.xticks(tickSep, labels=labelsArr)
    plt.yticks(tickSep, labels=labelsArr)
    plt.title(title)
    plt.colorbar()

    # Mutual info at bottom...
    plt.text(0.5, -0.1, f'Mutual Information: {mutual_info:.2f}', 
             horizontalalignment='center', 
             verticalalignment='top', 
             transform=plt.gca().transAxes, 
             fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig(save_path)
    plt.close()

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
    plt.plot(tmList, line, color='red', label=f'y={slope:.2f}x+{intercept:.2f}, $R^2$={r_value**2:.2f}')
    plt.legend()
    plt.title(title)
    plt.xlabel('TM Score')
    plt.ylabel('BLOSUM Score')
    plt.xlim(0.59,1)
    plt.ylim(-1,3)
    plt.show()


aaLetters=['A', 'R', 'N', 'D', 'C','Q','E','G','H','I','L', 'K', 'M', 'F', 'P', 'S', 'T','W', 'Y','V']
aaLettersTuples=[('A',), ('R',), ('N',), ('D',), ('C',),('Q',),('E',),('G',),('H',),('I',),('L',),('K',), ('M',),('F',), ('P',), ('S',), ('T',),('W',),('Y',),('V',),('X',)]

aaRotLetters_5PercentThreshold=[('A', '00'), ('R', '00'), ('R', '01'), ('R', '02'), ('R', '03'), ('N', '00'), ('N', '01'), ('N', '02'), ('N', '03'), ('N', '04'), ('N', '05'), ('N', '06'), ('D', '00'), ('D', '01'), ('D', '02'), ('D', '03'), ('C', '00'), ('C', '01'), ('C', '02'), ('Q', '00'), ('Q', '01'), ('Q', '02'), ('Q', '03'), ('Q', '04'), ('Q', '05'), ('E', '00'), ('E', '01'), ('E', '02'), ('E', '03'), ('E', '04'), ('E', '05'), ('E', '06'), ('G', '00'), ('H', '00'), ('H', '01'), ('H', '02'), ('H', '03'), ('H', '04'), ('H', '05'), ('H', '06'), ('I', '00'), ('I', '01'), ('I', '02'), ('I', '03'), ('L', '00'), ('L', '01'), ('K', '00'), ('K', '01'), ('K', '02'), ('K', '03'), ('M', '00'), ('M', '01'), ('M', '02'), ('M', '03'), ('M', '04'), ('M', '05'), ('M', '06'), ('F', '00'), ('F', '01'), ('F', '02'), ('F', '03'), ('P', '00'), ('P', '01'), ('S', '00'), ('S', '01'), ('S', '02'), ('T', '00'), ('T', '01'), ('T', '02'), ('W', '00'), ('W', '01'), ('W', '02'), ('W', '03'), ('W', '04'), ('W', '05'), ('Y', '00'), ('Y', '01'), ('Y', '02'), ('Y', '03'), ('Y', '04'), ('V', '00'), ('V', '01'), ('V', '02')]

backboneDihedralVals=[('S',),('E',),('-','-'),(0, 28), (0, 29), (1, 26), (1, 27), (1, 28), (1, 29), (2, 24), (2, 25), (2, 26), (2, 27), (2, 28), (2, 29), (3, 23), (3, 24), (3, 25), (3, 26), (3, 27), (3, 28), (3, 29), (4, 13), (4, 14), (4, 15), (4, 16), (4, 17), (4, 21), (4, 22), (4, 23), (4, 24), (4, 25), (4, 26), (4, 27), (4, 28), (4, 29), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 22), (5, 23), (5, 24), (5, 25), (5, 26), (5, 27), (5, 28), (5, 29), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 22), (6, 23), (6, 24), (6, 25), (6, 26), (6, 27), (6, 28), (6, 29), (7, 0), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 20), (7, 21), (7, 22), (7, 23), (7, 24), (7, 25), (7, 26), (7, 27), (7, 28), (7, 29), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 21), (8, 22), (8, 23), (8, 24), (8, 25), (8, 26), (8, 27), (8, 28), (8, 29), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 24), (9, 25), (9, 26), (9, 27), (9, 28), (9, 29), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 25), (10, 26), (10, 27), (10, 28), (11, 9), (11, 10), (11, 11), (11, 12), (11, 26), (19, 17), (19, 18), (19, 19), (20, 15), (20, 16), (20, 17), (20, 18), (21, 14), (21, 15), (21, 16), (22, 14), (22, 15), (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]
tickSepRegularBLOSUM=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

tickSepRot5PercentThreshold=[0,3,8,14,17,22,28,32,37,42,45,48,54,59,62,65,68,72,77,81]

#TM dictionary: dictionary of protein pair to TM score

centroidTuples=[(i,) for i in range(20)]

# Foldseek format
def writeFoldseekFormat(matrix, filename='mat3di.out'):
    # Define the header lines as specified
    header = '# 3Di bit/2'
    background = '# Background (precomputed optional): 0.0489372 0.0306991 0.101049 0.0329671 0.0276149 0.0416262 0.0452521 0.030876 0.0297251 0.0607036 0.0150238 0.0215826 0.0783843 0.0512926 0.0264886 0.0610702 0.0201311 0.215998 0.0310265 0.0295417 0.00001'
    lambda_value = '# Lambda     (precomputed optional): 0.351568'

    # List of amino acids including 'X' for the last row and column
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

    # Open the output file for writing
    with open(filename, 'w') as file:
        # Write the header lines to the file
        file.write(header + '\n')
        file.write(background + '\n')
        file.write(lambda_value + '\n')

        # Write the amino acid header line, separated by 3 spaces
        file.write('    ' + '   '.join(amino_acids) + '\n')

        # Write each row of the matrix
        for i, row in enumerate(matrix):
            # Add a 0 for the last column 'X'
            row.append(0)
            # Write the row label (amino acid) followed by the matrix values
            # Each value is formatted to be right-aligned within a 3-character wide field
            row_str = f"{amino_acids[i]:<2}" + " ".join(f"{value:>3}" for value in row)
            file.write(row_str + '\n')

        # Write the last row for 'X', filled with 0s
        last_row = "X " + " ".join(f"{0:>3}" for _ in range(21))
        file.write(last_row + '\n')
            
def dict_to_matrix(input_dict):
    # Initialize a 20x20 matrix with zeros
    matrix = [[0 for _ in range(20)] for _ in range(20)]
    
    # Populate the matrix with values from the dictionary
    for key, value in input_dict.items():
        indices = list(key)
        i = indices[0][0]
        j = indices[1][0] if len(indices) > 1 else indices[0][0]
        matrix[i][j] = value
        if i != j:
            matrix[j][i] = value
    
    return matrix

def getMutualInformation(data, labelsArr):
    
    singleCounts={}
    pairCounts={}

    # Initialize dictionaries with zeros
    for i in labelsArr:
        singleCounts[tuple(i)]=0
        for j in labelsArr:
            index=frozenset([tuple(i),tuple(j)])
            pairCounts[index]=0
    
    # Total counts for normalization...
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
    
    # print(totalSingleCount)
    # print(totalPairCount)

    # Convert counts to probabilities
    for key in singleCounts.keys():
        singleCounts[key]=singleCounts[key]/totalSingleCount
    for key in pairCounts.keys():
        pairCounts[key]=pairCounts[key]/totalPairCount

    # Calculate mutual information
    mutual_information = 0
    for pair, p_xy in pairCounts.items():

        # For the given JD, get the MDs:
        if len(pair) == 2:
            i1, i2 = pair
            p_x = singleCounts[tuple(i1)]
            p_y = singleCounts[tuple(i2)]
        else:
            i1, = pair
            p_x = p_y = singleCounts[tuple(i1)]

        # Omit value if any are 0...
        if p_x * p_y * p_xy != 0:
            mutual_information += p_xy * math.log(p_xy / (p_x * p_y)) # "Nats (natural log)"

    return mutual_information

# take CLI args for filename and name
import sys
filepath = sys.argv[1]
name = sys.argv[2]

with open(filepath, 'rb') as f:
    jesseCentroidDict=pickle.load(f)

myData = preprocessFileTM('protData/pairs_training.csv',jesseCentroidDict)
blosumJesse = makeBLOSUM(myData,centroidTuples)

np.save(f'{name}.npy', dict_to_matrix(blosumJesse))

writeFoldseekFormat(dict_to_matrix(blosumJesse), name + '.out')

MI = getMutualInformation(myData, centroidTuples)
print(f'Mutual Information: {MI}')

