import numpy as np
import random
from numba.typed import List as NumbaList
from numba import types, prange,njit,deferred_type,typed
from typing import List
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from collections import OrderedDict
from BPnumba.BPPdat import create_Bin,NumDBLF

sepecInd = OrderedDict()
sepecInd['fi'] = types.float64
sepecInd['genome'] = types.ListType(types.int64)
sepecInd['load'] = types.int64
@jitclass(sepecInd)
class Ind:
    def __init__(self,genome:List[int] ):
         self.fi = 0
         self.genome = genome
         self.load = 0
ind_type = deferred_type()
ind_type.define(Ind.class_type.instance_type)
@njit(nogil=True)
def create_intidivual(data:List[int]):
    return Ind(data)

@njit
def CreatePermutation(ls1:List[int])->List[int]:
    xmin =ls1[0]
    xmax= ls1[len(ls1)-1] 
    visited = [ False for i in np.arange(xmax+1)]
    newcode = np.zeros(xmax,dtype=np.int64)
    for i in np.arange(xmax):
        xj = random.randint(xmin,xmax)
        if xj == xmin:
            xj=xmin
            xmin +=1
        elif xj== xmax:
            xj=xmax
            xmax -=1
        while visited[xj]:
            xj = random.randint(xmin,xmax)
            if xj == xmin:
                xj=xmin
                xmin +=1
            elif xj == xmax:
                xj=xmax
                xmax -=1
        newcode[i] = xj
        visited[xj]=True
    return newcode
@njit
def CreatePoblation(num:int, ls2:List[int])->List[List[int]]:
    poblation = np.zeros(shape=(num, len(ls2)), dtype=np.int64)
    for _ in np.arange(num):
        poblation[_] = CreatePermutation(ls2)
    return poblation

@njit(nogil=True)
def CalcFi(ind:Ind, boxesData:List[List[int]], container:List[int]):
    bin = create_Bin(NumbaList(container))
    boxesData = NumbaList(boxesData)
    gene = ind.genome
    NumDBLF(bin,gene,boxesData)
    resp = bin.getLoadVol()/(container[0]*container[1]*container[2])
    ind.load=bin.getN()
    ind.fi = resp

@njit(parallel=True)
def InstancePob(pob:List[List[int]],boxesData:List[List[int]], container:List[int])->List[Ind]:
    lst = [ ]
    for i in prange(len(pob)):
        ind:Ind = create_intidivual(NumbaList(pob[i]))
        CalcFi(ind,boxesData,container)
        lst.append(ind)
    return lst


@njit
def Hamming(f1:List[int],f2:List[int])->float:
    count =0
    for i in np.arange(len(f1)):
        if f1[i] !=f2[i]:
            count +=1
    return np.float64(count)

@njit(nogil=True)
def Tournament(Pob:List[Ind],pt:float=0.85)->int:
    n = len(Pob)
    i1 = random.randrange(0, n)
    i2 = random.randrange(0, n) 
    while i1 == i2:
        i2 = random.randrange(0, n)
    r = random.random()
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
@njit
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
def InverseMutation(gen:List[int],i,j)->None:
    tmp =gen.copy()
    for k in np.arange(i,j+1):
        gen[k] = tmp[j-k+i]
@njit
def RandomSwapSeq(gen:List[int],index:int):
    cp = gen.copy()
    gen[:index]=cp[index+1:]
    gen[index+1:] = cp[:index]
@njit
def SwapSeqIndex(gen:List[int],index:int)->List[int]:
    n= len(gen)
    p1= gen[index+1:]
    p2 = gen[:index]
    l1 = len(p1)
    l2 = len(p2)
    nwgn = np.zeros(n,dtype=np.int64)
    for i in np.arange(l1):
        nwgn[i] = p1[i]
    nwgn[l1]=gen[index]
    for i in np.arange(l2):
        nwgn[l1+1+i] = p2[i]
    return nwgn
@njit
def Swap2Points(gen:List[int],i:int,j:int):
    auxpt = gen[i]
    gen[i] = gen[j]
    gen[j] = auxpt
@njit
def InsertionSeq(gen:List[int],indexToInsert:int,indexValue:int)->List[int]:
    newgen = gen.copy()
    newgen[indexToInsert]=gen[indexValue]
    for k in np.arange(indexToInsert,indexValue):
        newgen[k+1] = gen[k]
    return newgen
@njit
def InsertionSubSeq(gen:List[int],indexToInsert:int,i:int,j:int)->List[int]:
    subsq= gen[i:j+1]
    n = len(gen)
    valindx = gen[indexToInsert]
    nwgn = np.zeros(n,dtype=np.int64)
    visited = [ False for i in np.arange(len(gen)+1)]
    if indexToInsert + len(subsq)+1>n:
        dif = indexToInsert + len(subsq)-n
        indexToInsert -= dif
        indexToInsert -= 1
    l = indexToInsert
    for val in subsq:
        nwgn[l]=val
        visited[val] = True
        l+=1
    nwgn[l] = valindx
    visited[valindx] = True
    for k in np.arange(n):
        if nwgn[k] == 0:
            for v in np.arange(k,n):
                if not visited[gen[v]]:
                    nwgn[k] = gen[v]
                    visited[gen[v]]=True
                    break
    return nwgn
@njit
def RRSS(gen:List[int],index:int)->List[int]:
    n= len(gen)
    r1 = random.random()
    r2 = random.random()
    p1= gen[index+1:]
    p2 = gen[:index]
    if r1 > 0.5:
        p1 = p1[::-1]
    if r2 > 0.5:
        p2 = p2[::-1]
    l1 = len(p1)
    l2 = len(p2)
    nwgn = np.zeros(n,dtype=np.int64)
    for i in np.arange(l1):
        nwgn[i] = p1[i]
    nwgn[l1]=gen[index]
    for i in np.arange(l2):
        nwgn[l1+1+i] = p2[i]
    return nwgn
@njit
def RRIS(gen:List[int],indexToInsert:int,i:int,j:int)->List[int]:
    r = random.random()
    subsq= gen[i:j+1]
    if r > 0.5:
        subsq = subsq[::-1]
    n = len(gen)
    valindx = gen[indexToInsert]
    nwgn = np.zeros(n,dtype=np.int64)
    visited = [ False for i in np.arange(len(gen)+1)]
    if indexToInsert + len(subsq)+1>n:
        dif = indexToInsert + len(subsq)-n
        indexToInsert -= dif
        indexToInsert -= 1
    l = indexToInsert
    if r>0.5:
        nwgn[l] = valindx
        l+=1
    for val in subsq:
        nwgn[l]=val
        visited[val] = True
        l+=1
    if r <=0.5:
        nwgn[l] = valindx
    visited[valindx] = True
    for k in np.arange(n):
        if nwgn[k] == 0:
            for v in np.arange(k,n):
                if not visited[gen[v]]:
                    nwgn[k] = gen[v]
                    visited[gen[v]]=True
                    break
    return nwgn