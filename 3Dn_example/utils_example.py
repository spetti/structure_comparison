import numpy as np
import os
import bisect
import csv
import jax
import jax.numpy as jnp
from Bio import BiopythonDeprecationWarning
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq3
from Bio.PDB.Polypeptide import *

R_max=15
epsilon=.01*R_max #subtract from R_max for assignment
window=5 #discount neighbors within this quantity of index

N_R=5
N_theta=10
N_phi=5


r_intervals=np.zeros((N_R+1,))
phi_intervals=np.zeros((N_phi+1,))
theta_intervals=np.zeros((N_theta+1,))

V=4*(np.pi/3)*R_max**3


#split by equal volume
for i in range(N_R+1):
    r_intervals[i]=((V/N_R)*(3/(4*np.pi))*i)**(1/3)
# print(r_intervals)
for i in range(N_theta+1):
    theta_intervals[i]=2*np.pi*i/N_theta
# print(theta_intervals)
for i in range(N_phi+1):
    phi_intervals[i]=np.arccos(1-2*i/N_phi)
# print(phi_intervals)


def classifySecondaryStructure(phi,psi):
    if (-1.47 <= psi) & (psi <= 0.63) & (-2.51 <= phi) & (phi <= -0.42):
        return 'R'
    elif (-0.84 <= psi) & (psi <= 1.26) & (0.63 <= phi) & (phi <= 2.51):
        return 'L'
    elif ((0.63 <= psi) & (psi <= 3.14) & (-np.pi <= phi) & (phi <= -0.42)) or ((-np.pi <= psi) & (psi <= -2.09) & (-np.pi <= phi) & (phi <= -0.84)):
        return 'B'
    return 'X'
        
def determineBin(tup):
    binTup= (bisect.bisect_left(r_intervals,tup[0])-1,bisect.bisect_left(theta_intervals,tup[1])-1,bisect.bisect_left(phi_intervals,tup[2])-1)
    return (max(binTup[0],0), max(binTup[1],0), max(binTup[2],0))

#rotation matrices!
def rotMtxX(angle):
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
def rotMtxY(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
def rotMtxZ(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    
def cartesianToSpherical(point):
    r=np.sqrt(np.sum(np.square(point)))
    theta=np.arctan2(point[1],point[0])
    phi=np.arccos(point[2]/r)
    theta = theta + 2 * np.pi * (np.sign(theta) == -1)
    return (r,theta,phi)

def pw(x):
    '''compute pairwise distance'''
    x_norm = np.square(x).sum(-1)
    xx = np.einsum("...ia,...ja->...ij",x,x)
    sq_dist = x_norm[...,:,None] + x_norm[...,None,:] - 2 * xx
    return np.sqrt(sq_dist+ 1e-10)

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
        return 'X','X'
    

def generateBinList(N_R=5,N_Theta=10,N_Phi=5):
    allBins=[]
    for a in ['B','L','R','X']:
        for i in range(N_R):
            for j in range(N_theta):
                for k in range(N_phi):
                    allBins.append((a,i,j,k))
    return allBins

def redefineGeoSystem(C_alpha,C_beta,N):
    C_beta=C_beta-C_alpha
    N=N-C_alpha
    
    angleToXZPlane=np.arctan2(C_beta[1],C_beta[2])
    rotMtx1=rotMtxX(angleToXZPlane)
    
    C_beta=rotMtx1@C_beta
    N=rotMtx1@N
    
    angleToZ=np.arctan2(-C_beta[0],C_beta[2])
    rotMtx2=rotMtxY(angleToZ)
    
    C_beta=rotMtx2@C_beta
    N=rotMtx2@N
    
    angleToXZPlane=np.arctan(-N[1]/N[0])
    angleToXZPlane_alt=np.pi+angleToXZPlane
    
    case=(N[0]*np.cos(angleToXZPlane)-N[1]*np.sin(angleToXZPlane))>0
    
    if case!=True:
        angleToXZPlane=angleToXZPlane_alt
    rotMtx3=rotMtxZ(angleToXZPlane)
    
    return rotMtx3@(rotMtx2@rotMtx1)

def allBinsNeighborsPerProt(prot1,R_max=15):
    allBinsList=[]
    allCoordCA,allCoordCB,allCoordN,allSecondaryStruct=getAllPositions(prot1)
    dists,neighborMtx=closestNeighbors(allCoordCA,R_max-epsilon,window)
    numOfAA=len(dists)
    
    for aa in range(numOfAA):
        neighborBinList=[]
        curNeighborList=neighborMtx[aa,:]
        neighbors=np.where(curNeighborList==1)
        
        C_alpha=allCoordCA[aa]
        C_beta=allCoordCB[aa]
        N=allCoordN[aa]
        rotMtxForAA=redefineGeoSystem(C_alpha,C_beta,N)
        
        for j in neighbors[0]:
            #point needs to be in coord system s.t. C_alpha is origin
            point=allCoordCA[j]-C_alpha
            pointAdjusted=rotMtxForAA@point
            sphericalCoord=cartesianToSpherical(pointAdjusted)
            sphericalBins=determineBin(sphericalCoord)
            binTuple=(j,allSecondaryStruct[j],)+sphericalBins
            neighborBinList.append(binTuple)
        allBinsList.append(neighborBinList)
    return allBinsList

#for a protein, returns all CA, CB. -1 if doesn't exist (e.g. Gly)
def getAllPositions(prot1): 
    allCoordCA=[]
    allCoordCB=[]
    allCoordN=[]
    allSecondaryStruct=[]
    fileName=prot1
    p = PDBParser()
    structure = p.get_structure("my_protein", fileName)
    
    for model in structure:
        for chain in model:
            residues=list(chain)
            for i in range(len(residues)):
                residue=residues[i]
                phi='X'
                psi='X'
                if residue.get_resname()!='GLY':
                    try: 
                        allCoordCA.append(np.array(residue['CA'].get_coord()))
                        allCoordCB.append(np.array(residue['CB'].get_coord()))
                        allCoordN.append(np.array(residue['N'].get_coord()))
                    except KeyError:
                        allCoordCA.append('X')
                        allCoordCB.append('X')
                        allCoordN.append('X')
                else:
                    try:
                        allCoordCA.append(np.array(residue['CA'].get_coord()))
                        #equation from: https://github.com/gjoni/trDesign/blob/master/01-hallucinate/src/utils.py
                        Ca=residue['CA'].get_coord()
                        N=residue['N'].get_coord()
                        C=residue['C'].get_coord()
                        b = Ca - N
                        c = C - Ca
                        a = np.cross(b, c)
                        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
                        allCoordCB.append(np.array(Cb))
                        allCoordN.append(np.array(residue['N'].get_coord()))
                    except KeyError:
                        allCoordCA.append('X')
                        allCoordCB.append('X')
                        allCoordN.append('X')
                if i==0 or i==len(residues)-1:
                    allSecondaryStruct.append('X')
                else:
                    phi, psi = calc_phi_psi(residues[i - 1], residues[i], residues[i + 1])
                    allSecondaryStruct.append(classifySecondaryStructure(phi,psi))
    return allCoordCA, allCoordCB, allCoordN, allSecondaryStruct

def closestNeighbors(allCoordCA,distThreshold,adjThreshold):
    dists=pw(allCoordCA)
    neighborMtx = np.where(dists < distThreshold, 1, 0)
    
    for i in range(len(neighborMtx)):
        neighborMtx[i,max(i-adjThreshold,0):(min(i+adjThreshold+1,len(neighborMtx)))]=0
        
    return dists,neighborMtx

def getIndex(l,value):
    try: 
        return l.index(value)
    except: 
        return -1
    
    
def oneHotFromListRowVec(l,mtxIndex):
    binCt=len(mtxIndex)
    oneHot=np.zeros((binCt))
    dim=len(l)
    for j in range(dim):
        binVal=l[j][1:]
        ind1=getIndex(mtxIndex,binVal)
        oneHot[ind1]+=1
    return oneHot

def getOneHot(nbrList_All,mtxIndex):
    if -1 not in mtxIndex:
        mtxIndex.append(-1)
    length=len(nbrList_All)
    binCt=len(mtxIndex)
    vec=np.zeros((length,binCt),dtype=int)
    
    for j in range(length):
        currentAA=nbrList_All[j]
        vec[j,:]=np.array(oneHotFromListRowVec(currentAA,mtxIndex))
        
    vec[:,-1]=5
    return vec

def getOneHot_Clustered(protClustList):
    labels=[(i,) for i in range(20)]
    length=len(protClustList)
    vec=np.zeros((length,len(labels)),dtype=int)
    for j in range(length):
        ind=getIndex(labels,protClustList[j])
        vec[j,ind]=1
    return vec
    
def sim_mtx(oh_seq1, oh_seq2, blosum):
    return np.einsum('ij,jk,lk->il', oh_seq1, blosum, oh_seq2) 

def jaccard1D(v1,v2):
    min_elements = np.minimum(v1,v2)
    max_elements = np.maximum(v1,v2)
    
    # Sum over the last dimension to get the final result
    result = np.sum(min_elements)/np.sum(max_elements)
    
    return 1-result

def find_closest_centroid(vector, centroids):
    distances = [jaccard1D(vector, centroid) for centroid in centroids]
    return np.argmin(distances)

def seqFromCentroids(prot_bv,centroids):
    centSeq=[]
    for row in prot_bv:
        centSeq.append((find_closest_centroid(row,centroids),))
    return centSeq


def sw_affine(
    restrict_turns=True,
    penalize_turns=True,
    batch=True,
    unroll=2,
    NINF=-1e30
):
    """smith-waterman (local alignment) with affine gap"""
    # rotate matrix for vectorized dynamic-programming

    def rotate(x):
        # solution from jake vanderplas (thanks!)
        a, b = x.shape
        ar, br = jnp.arange(a)[::-1, None], jnp.arange(b)[None, :]
        i, j = (br - ar) + (a - 1), (ar + br) // 2
        n, m = (a + b - 1), (a + b) // 2
        output = {
            "x": jnp.full([n, m], NINF).at[i, j].set(x),
            "o": (jnp.arange(n) + a % 2) % 2
        }
        return output, (jnp.full((m, 3), NINF), jnp.full((m, 3), NINF)), (i, j)

    # fill the scoring matrix
    def sco(x, lengths, gap=0.0, open=0.0, temp=1.0):

        def _soft_maximum(x, axis=None, mask=None):
            def _logsumexp(y):
                y = jnp.maximum(y, NINF)
                if mask is None:
                    return jax.nn.logsumexp(y, axis=axis)
                else:
                    return (y.max(axis)
                            + jnp.log(jnp.sum(
                                mask * jnp.exp(
                                    y - y.max(axis, keepdims=True)
                                ),
                                axis=axis
                            )))
            return temp * _logsumexp(x / temp)

        def _cond(cond, true, false):
            return cond * true + (1 - cond) * false

        def _pad(x, shape):
            return jnp.pad(x, shape, constant_values=(NINF, NINF))

        def _step(prev, sm):
            h2, h1 = prev   # previous two rows of scoring (hij) mtxs

            Align = jnp.pad(h2, [[0, 0], [0, 1]]) + sm["x"][:, None]
            Right = _cond(
                sm["o"],
                _pad(h1[:-1], ([1, 0], [0, 0])),
                h1
            )
            Down = _cond(
                sm["o"],
                h1,
                _pad(h1[1:], ([0, 1], [0, 0]))
            )

            # add gap penalty
            if penalize_turns:
                Right += jnp.stack([open, gap, open])
                Down += jnp.stack([open, open, gap])
            else:
                gap_pen = jnp.stack([open, gap, gap])
                Right += gap_pen
                Down += gap_pen

            if restrict_turns:
                Right = Right[:, :2]

            h0_Align = _soft_maximum(Align, -1)
            h0_Right = _soft_maximum(Right, -1)
            h0_Down = _soft_maximum(Down, -1)
            h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1)
            return (h1, h0), h0

        # mask
        a, b = x.shape
        real_a, real_b = lengths
        mask = (
            (jnp.arange(a) < real_a)[:, None] *
            (jnp.arange(b) < real_b)[None, :]
        )
        x = x + NINF * (1 - mask)

        sm, prev, idx = rotate(x[:-1, :-1])
        hij = jax.lax.scan(
            _step,
            prev,
            sm,
            unroll=unroll
        )[-1][idx]

        # sink
        return _soft_maximum(
            hij + x[1:, 1:, None],
            mask=mask[1:, 1:, None]
        )

    # traceback to get alignment (aka. get marginals)
    traceback = jax.grad(sco)

    # add batch dimension
    if batch:
        return jax.vmap(traceback, (0, 0, None, None, None))
    else:
        return traceback
    
def lddt2(coord_1, coord_2, aln, query_length):
    #n as number of aligned positions
    aligned = aln > 0.95

    n = int(jnp.sum(aligned))
    # Compute pw distances; however many are under 15 in structure 1 is the denom
    pw_1= pw(coord_1)
    mask = (jnp.arange(pw_1.shape[0]) < query_length)[:,None] * (jnp.arange(pw_1.shape[0]) < query_length)[None,:]
    denom =4*(jnp.sum(jnp.where(pw_1 <15,1,0)*mask)- query_length)/2
    
    # Reduce so that only aligned positions appear first
    row_indices, col_indices = jnp.where(aln > 0.95, size = pw_1.shape[0], fill_value =0) #will have length coord_1
    reduced_coord_1 = jnp.zeros_like(coord_1)
    reduced_coord_1 = reduced_coord_1.at[:len(row_indices)].set(jnp.take(coord_1, row_indices, axis=0))
    reduced_coord_2 = jnp.zeros_like(coord_1) # only need to be min of the two seq lengths; so ok to do coord_1 shape
    reduced_coord_2 = reduced_coord_2.at[:len(col_indices)].set(jnp.take(coord_2, col_indices, axis=0))

    # Compute distance differences
    pw_1 = pw(reduced_coord_1)
    pw_2 = pw(reduced_coord_2)
    
    # Mask restricts to pairs we care about; n is number of aligned positions and we have permuted so these are the first n
    mask = (jnp.arange(pw_1.shape[0]) < n)[:,None] * (jnp.arange(pw_1.shape[0]) < n)[None,:]
    mask *= jnp.triu(jnp.ones_like(pw_1), k=1)
    mask *= (pw_1<15)

    # Count how many under thresholds
    distance_diffs = jnp.abs(pw_1-pw_2)

    num = jnp.sum((distance_diffs<0.5)*mask) + jnp.sum((distance_diffs<1.0)*mask) + jnp.sum((distance_diffs<2.0)*mask) + jnp.sum((distance_diffs<4.0)*mask) 

    #return num/denom, unless denom is zero
    return jnp.where(denom != 0, num / denom, 0)

def distMtx(allCoordCA,distThreshold):
    dists=pw(allCoordCA)
    return dists

def reducedDistMtx(alnMtx,distMtx1,distMtx2):
    alnMtx=np.where(alnMtx>0.95,1,0)
    dim=np.sum(alnMtx)
    dim=int(dim)
    redDistMtx1=np.zeros((dim,dim))
    redDistMtx2=np.zeros((dim,dim))
    list1=[]
    list2=[]
    for i in range(np.shape(alnMtx)[0]):
        for j in range(np.shape(alnMtx)[1]):
            if alnMtx[i,j]==1:
                list1.append(i)
                list2.append(j)

    for l in range(len(list1)):
        for k in range(len(list1)):
            redDistMtx1[l,k]=distMtx1[list1[l],list1[k]]
            redDistMtx2[l,k]=distMtx2[list2[l],list2[k]]
    return redDistMtx1,redDistMtx2



#does prot1 as ref
def detLDDT(CA_prot1,CA_prot2,aln):
    
    
    distMtx1=distMtx(CA_prot1,15)
    distMtx2=distMtx(CA_prot2,15)
    
    reducedDistMtx1,reducedDistMtx2=reducedDistMtx(aln,distMtx1,distMtx2)
    
    thresh=[.5,1,2,4]
    threshV=[0,0,0,0]
    
    mask=np.where(reducedDistMtx1<15,1,0)-np.eye(np.shape(reducedDistMtx1)[0])
    
    
    relevantDist=(abs(reducedDistMtx1-reducedDistMtx2))
    for i in range(len(thresh)):
        threshV[i]=np.sum(np.where(relevantDist<thresh[i],1,0)*mask)

            
    denom=4*np.sum(np.where(distMtx1<15,1,0)-np.eye(np.shape(distMtx1)[0]))
    num=np.sum(np.array(threshV))
    
    if denom==0:
        return 0
    return num/denom
                