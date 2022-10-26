import numpy as np
import random
from numba import types, njit
from typing import List
from numba.typed import List as NumbaList
from BPnumba.BoxN import ItemBin
from BPnumba.BPPH import create_Bin,DBLF,DBLF2
from BPnumba.Individual import Ind

@njit
def Hamming(f1:Ind,f2:Ind):
    count =0
    N=len(f1.genome)
    for i in np.arange(N):
        if f1.genome[i] != f2.genome[i] and f1.genome_r[i] != f2.genome_r[i]:
            count +=1
    return count
    
@njit #Cruza por orden
def CrossOX(P1:List[int],P2:List[int],R1:List[int],R2:List[int],i:int,j:int)->List[List[int]]:
    n = len(P1)
    h1 = np.zeros(n,dtype=np.int64)
    rh1 = np.zeros(n,dtype=np.int64)
    visited =  [False for i in np.arange(n+1)]
    for k in np.arange(i,j+1):
        h1[k]=P1[k]
        rh1[k] = R1[k]
        visited[P1[k]] = True
    for k in np.arange(j+1,n):
        for l in np.arange(0,n):
            if not visited[P2[l]]:
                h1[k] = P2[l]
                rh1[k] = R2[l]
                visited[P2[l]] = True
                break
    for k in np.arange(0,i):
        for l in np.arange(0,n):
            if not visited[P2[l]]:
                visited[P2[l]] = True
                h1[k] = P2[l]
                rh1[k] = R2[l]
                break
    return [h1,rh1]
@njit
def SwapPointValue(gen:List[int],geneR:List[int], i:int,val:int)->List[List[int]]:
    if gen[i] == val:
        return
    tmp = gen[i]
    tmpr = geneR[i]
    n = len(gen)
    for j in np.arange(n):
        if i!=j and gen[j]==val:
            gen[i] = val
            geneR[i] = geneR[j]
            gen[j]=tmp
            geneR[j] = tmpr
            break
@njit #InverseMutation o 2-OPT mutation
def InverseMutation(gen:List[int],rot:List[int],i:int,j:int)->List[List[int]]: #Random reversing of subsequence (RRS)
    tmp =gen.copy()
    tmp2 = rot.copy()
    tmp[i:j+1] = gen[i:j+1][::-1]
    tmp2[i:j+1] = rot[i:j+1][::-1]
    return [tmp,tmp2]  


@njit
def RI(gen:List[int],i:int,J:int): #Random Insertion
    n=len(gen)
    valueToInsert = gen[J]
    if i != 0:
        tmp=gen[:i]
    tmp.append(valueToInsert)
    tmp[i+1:J+1] = gen[i:J]
    if J!=n-1:
        tmp[J+1:] = gen[J+1:]
    return tmp
@njit
def RIS(gen:List[int],indexToInsert:int,i:int,j:int)->List[int]: #Random Insertion of subsequence
    subsq= gen[i:j+1]
    n = len(gen)
    valindx = gen[indexToInsert]
    if indexToInsert < i:
        nwgn = gen[:indexToInsert]
        nwgn.extend(subsq)
        nwgn.append(valindx)
        if indexToInsert < i-1:
            nwgn.extend(gen[indexToInsert+1:i])
        if j<n-1:
            nwgn.extend(gen[j+1:])
    elif indexToInsert > j:
        nwgn=gen[:i]
        if j+1<indexToInsert:
            nwgn.extend(gen[j+1:indexToInsert])
        if indexToInsert < n-1: #si lo dejo sin esta condicion, en este caso, la secuencia que igual
            nwgn.extend(subsq)
            nwgn.append(valindx)
        else:
            nwgn.append(valindx)
            nwgn.extend(subsq)
        nwgn.extend(gen[indexToInsert+1:])
    if len(nwgn)!=n:
        raise Exception("Error in Genetic Operator: RIS")
    return nwgn
@njit
def RRIS(gen:List[int],indexToInsert:int,i:int,j:int)->List[int]: #Random reversing insertion of subsequence
    r = np.random.random()
    subsq= gen[i:j+1]
    if r > 0.5:
        subsq = subsq[::-1]
    n = len(gen)
    valindx = gen[indexToInsert]
    if indexToInsert < i:
        nwgn = gen[:indexToInsert]
        nwgn.extend(subsq)
        nwgn.append(valindx)
        if indexToInsert < i-1:
            nwgn.extend(gen[indexToInsert+1:i])
        if j<n-1:
            nwgn.extend(gen[j+1:])
    elif indexToInsert >j:
        nwgn=gen[:i]
        if j+1<indexToInsert:
            nwgn.extend(gen[j+1:indexToInsert])
        if indexToInsert < n-1 or r>0.5:
            nwgn.extend(subsq)
            nwgn.append(valindx)
        else:
            nwgn.append(valindx)
            nwgn.extend(subsq)
        nwgn.extend(gen[indexToInsert+1:])
    if len(nwgn)!=n:
        raise Exception("Error in Genetic Operator:RRIS")
    return nwgn


@njit
def Combine2(r:float,gen:List[int],indexToInsert:int,i:int,j:int):
    if r <= 1/3:
        return RI(gen,i,j)
    elif r>1/3 and r<=2/3:
        return RIS(gen,indexToInsert,i,j)
    else:
        return RRIS(gen,indexToInsert,i,j)
@njit
def MutateC2(genome: List[int],rgenome: List[int], randomStep: int=-1)->List[List[int]]:
    n = int(len(genome))
    step=randomStep
    r = np.random.random()
    if randomStep==-1:
        step = np.random.randint(1,n-2)
    elif randomStep<2 and randomStep>=0:
        step=2
    init = np.random.randint(1,int(n/2))
    if init + step > n-2:
        end = n-2
    else:
        end =  init+step-1
    if np.random.random()<0.5:
        index=np.random.randint(0,init)
    else:
        index=np.random.randint(end+1,n)
    gene = Combine2(r,genome,index,init,end)
    rgene = Combine2(r,rgenome,index,init,end)
    return [gene,rgene]



@njit
def RS(gen:List[int],i:int,j:int)->List[int]: #Random Swap
    tmp = gen.copy()
    auxpt = tmp[i]
    tmp[i] = tmp[j]
    tmp[j] = auxpt
    return tmp
@njit
def RSS(gen:List[int],ini1:int,end1:int,ini2:int,end2:int)->List[int]:#Random Swap Subsequences
    if end1-ini1 != end2-ini2:
        raise Exception("Subsequences most be with same lenght.")
    tmp = gen.copy()
    tmp[ini1:end1+1] = gen[ini2:end2+1]
    tmp[ini2:end2+1] = gen[ini1:end1+1]
    return tmp

@njit
def RRSS(gen:List[int],ini1:int,end1:int,ini2:int,end2:int,r1:float,r2:float)->List[int]: #Random reversing swap of subsequences
    n= len(gen)
    if end1-ini1 != end2-ini2:
        raise Exception("Subsequences most be with same lenght.")
    tmp = gen.copy()

    if r1 <= 0.5:
        tmp[ini1:end1+1] = gen[ini2:end2+1][::-1]
    else:
        tmp[ini1:end1+1] = gen[ini2:end2+1]
    if r2<= 0.5:
        tmp[ini2:end2+1] = gen[ini1:end1+1][::-1]
    else:
        tmp[ini2:end2+1] = gen[ini1:end1+1]
    return tmp
@njit
def Combine1(r:float,gen:List[int],i:int,j:int,i2:int,j2:int,r1:float,r2:float)->List[int]:
    if r <= 1/3:
        return RS(gen,i,j2)
    elif r>1/3 and r<=2/3:
        return RSS(gen,i,j,i2,j2)
    else:
        return RRSS(gen,i,j,i2,j2,r1,r2)

@njit
def MutateC1(genome:List[int],rgenome: List[int],randomStep:int=-1)->List[List[int]]:
    n=len(genome)
    step=randomStep
    r = np.random.random()
    if randomStep==-1:
        step = np.random.randint(1,n-2)
    elif randomStep<2 and randomStep>=0:
        step=2
    i=np.random.randint(int(n/2)-int(step/2)+1)
    j = i + int(step/2)-1
    i2 = np.random.randint(int(n/2),n-int(step/2))
    j2 = i2 + int(step/2)-1
    
    r1 = np.random.random()
    r2 = np.random.random()
    gene = Combine1(r,genome,i,j,i2,j2,r1,r2)
    rgene = Combine1(r,rgenome,i,j,i2,j2,r1,r2)
    return [gene,rgene]

@njit
def MutateInversion(gene:List[int],rotGene:List[int],randomStep:int=-1)->List[List[int]]:
    n=len(gene)
    step=randomStep
    if randomStep==-1:
        step = np.random.randint(1,n-2)
    elif randomStep<2:
        step=2
    i= random.randrange(1,int(n/2))
    j= i+step-1
    return InverseMutation(NumbaList(gene),rotGene,i,j)