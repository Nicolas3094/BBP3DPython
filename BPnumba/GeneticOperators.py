from operator import ge, index
import numpy as np
import random
from numba.typed import List as NumbaList
from numba import types, njit,deferred_type
from typing import List
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from collections import OrderedDict
from BPnumba.BPPH import create_Bin,DBLF,DBLF2

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

def CreateHeuristicPob(num:int,BoxSeq:list,reverse=False):
    poblation = []
    originalInd = np.arange(1,len(BoxSeq)+1,dtype=np.int64)
    DataSet = list(zip(originalInd, BoxSeq))
    DataSet.sort(key=lambda x : x[1][0]*x[1][1]*x[1][2], reverse=reverse)
    volSeq = np.array([vec[0] for vec in DataSet],dtype=np.int64)
    poblation.append(volSeq)
    for i in range(3):  # ordena por longitud, ancho y alto
        DataSet.sort(key=lambda x : x[1][i], reverse=reverse)
        di = np.array([vec[0] for vec in DataSet],dtype=np.int64)
        poblation.append(di)
    for i in np.arange(num-4):
        p2 = CreatePermutation(originalInd)
        poblation.append(p2)
    return np.asarray(poblation)

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

@njit #InverseMutation o 2-OPT mutation
def InverseMutation(gen:List[int],i:int,j:int)->List[int]:
    tmp =gen.copy()
    tmp[i:j+1] = gen[i:j+1][::-1]
    return tmp
@njit
def RandomSwapSeq(gen:List[int],index:int)->List[int]:
    cp = gen.copy()
    cp[:index]=gen[index+1:]
    cp[index+1:] = gen[:index]
    return cp

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
    r1 = np.random.random()
    r2 = np.random.random()
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
        raise Exception("Error in Genetic Operator")
    return nwgn