import numpy as np
import random
from numba import types, njit,deferred_type,objmode, prange
from typing import List
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from collections import OrderedDict
from BPnumba.BPPH import create_Bin,DBLF,DBLF2
import time


sepecInd = OrderedDict()
sepecInd['fi'] = types.float64
sepecInd['genome'] = types.ListType(types.int64)
sepecInd['load'] = types.int64
sepecInd['BinBoxes'] = types.ListType(types.int64)
sepecInd['codeSolution'] =  types.string
@jitclass(sepecInd)
class Ind:
    def __init__(self,genome:List[int] ):
         self.fi:float = 0
         self.genome = genome
         self.load = 0
         self.BinBoxes = NumbaList(np.array([0],dtype=np.int64))
         self.codeSolution = ''
ind_type = deferred_type()
ind_type.define(Ind.class_type.instance_type)

@njit
def create_intidivual(gen:List[int]):
    return Ind(gen)

@njit 
def CodeSolution(idLoaded:List[int])->types.unicode_type:
    st = "|"
    for i in np.arange(len(idLoaded)):
        st += str(idLoaded[i])+"|"
    return st

@njit
def CreatePermutation(ls1:List[int])->List[int]:
    return np.asarray(np.random.choice(ls1,len(ls1), replace=False),dtype=np.int64)
@njit #(nogil=True)
def CreateHeuristicPob(num:int,Data:List[List[int]],bin:List[int],reversed=True)->List[List[int]]:
    Ub = len(Data)
    seq=np.arange(1,Ub+1)
    convertedData= [ (Data[i],i+1) for i in np.arange(Ub,dtype=np.int64)]
    
    poblation:list[list[int]] = []
    
    convertedData.sort(key=lambda x:x[0][0]*x[0][1]*x[0][2],reverse=reversed)
    nwsq = []
    for i in np.arange(Ub):
       nwsq.append(convertedData[i][1])
    poblation.append(nwsq)

    convertedData.sort(key=lambda x:x[0][0],reverse=reversed)
    nwsq = []
    for i in np.arange(Ub):
       nwsq.append(convertedData[i][1])
    poblation.append(nwsq)

    convertedData.sort(key=lambda x:x[0][1],reverse=reversed)
    nwsq = []
    for i in np.arange(Ub):
       nwsq.append(convertedData[i][1])
    poblation.append(nwsq)

    convertedData.sort(key=lambda x:x[0][2],reverse=reversed)
    nwsq = []
    for i in np.arange(Ub):
       nwsq.append(convertedData[i][1])
    poblation.append(nwsq)
    
    for _ in np.arange(num-4):
        p = list(CreatePermutation(seq))
        poblation.append(p)
    return poblation

@njit
def CreatePoblation(num:int, ls2:List[int])->List[List[int]]:
    poblation = np.zeros(shape=(num, len(ls2)), dtype=np.int64)
    for _ in np.arange(num):
        poblation[_] = CreatePermutation(ls2)
    return poblation

@njit
def CalcFi(ind:Ind, boxesData:List[List[int]], container:List[int],heuristic:int=0):
    bin = create_Bin(NumbaList(container))
    boxesData = NumbaList(boxesData)
    gene = ind.genome
    if heuristic == 0:
        DBLF(bin,gene,boxesData)
    else:
        DBLF2(bin,gene,boxesData)
    resp = bin.getLoadVol()/(container[0]*container[1]*container[2])
    ind.load=bin.getN()
    ind.fi = resp
    ind.codeSolution = CodeSolution(bin.getBoxes())

@njit
def InstancePob(pob:List[List[int]],boxesData:List[List[int]], container:List[int],heuristic:int=0)->List[Ind]:
    lst = [ ]
    for i in np.arange(len(pob)):
        ind:Ind = create_intidivual(NumbaList(pob[i]))
        CalcFi(ind,boxesData,container,heuristic)
        lst.append(ind)
    return lst

@njit
def Hamming(f1:List[int],f2:List[int])->float:
    count =0
    for i in np.arange(len(f1)):
        if f1[i] !=f2[i]:
            count +=1
    return np.float64(count)

@njit
def Tournament(Pob:List[Ind],pt:float=1.0)->int:
    n = len(Pob)
    i1 = np.random.randint(0, n)
    i2 = np.random.randint(0, n) 
    while i1 == i2:
        i2 = np.random.randint(0, n)
    r = np.random.random()
    if r <= pt:
        if Pob[i1].fi > Pob[i2].fi:
            return i1
        else:
            return i2
    else:
        if Pob[i1].fi > Pob[i2].fi:
            return i2
        else:
            return i1
@njit
def RouletteWheel(Pob:List[Ind])->int:
    pobFi = np.array([ ind.fi  for ind in Pob])
    totalFi = np.sum(pobFi)
    pobFi /= totalFi
    init = np.random.randint(len(pobFi))
    r = np.random.random()
    suma = 0
    i=init
    while suma < r:
        suma += pobFi[i]
        if suma >=r:
            break
        if i == len(pobFi)-1:
            i=-1
        i+=1
    return i
    
@njit #Cruza por orden
def CrossOX(P1:List[int],P2:List[int],i:int,j:int)->List[int]:
    n = len(P1)
    h1 = np.zeros(n,dtype=np.int64)
    visited =  [False for i in np.arange(n+1)]
    for k in np.arange(i,j+1):
        h1[k]=P1[k]
        visited[P1[k]] = True
    for k in np.arange(j+1,n):
        for l in np.arange(0,n):
            if not visited[P2[l]]:
                h1[k] = P2[l]
                visited[P2[l]] = True
                break
    for k in np.arange(0,i):
        for l in np.arange(0,n):
            if not visited[P2[l]]:
                visited[P2[l]] = True
                h1[k] = P2[l]
                break
    return h1
@njit
def SwapPointValue(gen:List[int], i:int,val:int):
    if gen[i] == val:
        return
    tmp = gen[i]
    n = len(gen)
    for j in np.arange(n):
        if i!=j and gen[j]==val:
            gen[i] = val
            gen[j]=tmp
            break
@njit #InverseMutation o 2-OPT mutation
def InverseMutation(gen:List[int],i:int,j:int)->List[int]: #Random reversing of subsequence (RRS)
    tmp =gen.copy()
    tmp[i:j+1] = gen[i:j+1][::-1]
    return tmp   

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
def RRSS(gen:List[int],ini1:int,end1:int,ini2:int,end2:int)->List[int]: #Random reversing swap of subsequences
    n= len(gen)
    if end1-ini1 != end2-ini2:
        raise Exception("Subsequences most be with same lenght.")
    tmp = gen.copy()
    r1 = np.random.random()
    r2 = np.random.random()
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
def Combine1(gen:List[int],i:int,j:int,i2:int,j2:int):
    r = np.random.random()
    if r <= 1/3:
        return RS(gen,i,j2)
    elif r>1/3 and r<=2/3:
        return RSS(gen,i,j,i2,j2)
    else:
        return RRSS(gen,i,j,i2,j2)

@njit
def Combine2(gen:List[int],indexToInsert:int,i:int,j:int):
    r = np.random.random()
    if r <= 1/3:
        return RI(gen,i,j)
    elif r>1/3 and r<=2/3:
        return RIS(gen,indexToInsert,i,j)
    else:
        return RRIS(gen,indexToInsert,i,j)
