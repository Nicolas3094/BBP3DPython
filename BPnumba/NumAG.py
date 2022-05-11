import numpy as np
import random
from BPnumba.GOnumba import Hamming
from numba.typed import List as NumbaList
from numba import types, prange,njit,deferred_type
from typing import List
from collections import OrderedDict
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from BPnumba.GOnumba import InverseMutation
from BPnumba.NumFun import create_Bin,NumDBLF

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
def InstancePob(pob:List[List[int]],boxesData:List[List[int]], container:List[int]):
    lst = [ ]
    for i in prange(len(pob)):
        ind:Ind = create_intidivual(NumbaList(pob[i]))
        CalcFi(ind,boxesData,container)
        lst.append(ind)
    return lst
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
@njit(nogil=True)
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
@njit(nogil=True)
def Tournament(Pob:List[ind_type],pt:float=0.85)->int:
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
specAG = OrderedDict()
specAG['_prSelect'] = types.float64
specAG['_prMut'] = types.float64
specAG['_prCross'] = types.float64
@jitclass(specAG)
class NAG:
    def __init__(self,ps:float,pm:float,pc:float):
        self._prSelect=ps
        self._prMut =pm
        self._prCross=pc
    @staticmethod
    def Selection(poblation:List[Ind],pr_selecton:float)->int:
        return Tournament(poblation,pr_selecton)
    @staticmethod
    def Crossover(ind1:Ind,ind2:Ind)->List[int]:
        r = random.random()
        n = len(ind1.genome)
        i= random.randrange(3,int(n/2))
        j= random.randrange(i+1,n)
        resp = CrossOX(ind1.genome,ind2.genome,i,j)
        return resp
    @staticmethod
    def Mutation(gene:List[int],prmUt):
        r = random.random()
        if r <=  prmUt:
            n = len(gene)
            i= random.randrange(1,int(n/2))
            j= random.randrange(i+1,n)
            InverseMutation(NumbaList(gene),i,j)
    @staticmethod
    def Elitism(pob:List[Ind],bestNum:int)->List[Ind]:
        return pob[:bestNum]       
        
@njit(nogil=True)
def create_AG(ps:float,pm:float,pc:float):
    return NAG(ps,pm,pc)
ag_type = deferred_type()
ag_type.define(NAG.class_type.instance_type)

@njit(parallel=True)
def CreateNexGen(GA:NAG,pob:List[Ind],datos,contenedor):
    n = len(pob)
    k = random.randint(int(n/4),n) #numero de individuos intentar por crear por pares
    existedPob = [ind.genome for ind in pob]
    if k % 2 != 0:
         k -=1 
    for _ in np.arange(k):
        id1 = GA.Selection(pob,GA._prSelect)
        id2 = GA.Selection(pob,GA._prSelect)
        while id2 == id1:
            id2 = GA.Selection(pob,GA._prSelect)
        rn = random.random()
        if rn <= GA._prCross:
            rp = random.random()
            if GA._prMut<=rp:
                h1 = GA.Crossover(pob[id1],pob[id2])
                h2 = GA.Crossover(pob[id2],pob[id1])
                foo =True
                foo2=True
                for j in prange(len(existedPob)):
                    if Hamming(NumbaList(h1),existedPob[j]) == 0 and foo:
                        foo = False
                        break
                for j in prange(len(existedPob)):
                    if Hamming(NumbaList(h2),existedPob[j]) == 0 and foo2:
                        foo2 = False
                        break                        
                if foo:
                    ind1 = create_intidivual(NumbaList(h1))
                    CalcFi(ind1,NumbaList(datos),NumbaList(contenedor))
                    pob.append(ind1)
              
                if foo2:
                    ind2 = create_intidivual(NumbaList(h2))
                    CalcFi(ind2,NumbaList(datos),NumbaList(contenedor))
                    pob.append(ind2)
        pob.sort(key=lambda ind : ind.fi, reverse=True)
@njit
def TrainAG(maxItr:int,pob:List[Ind],pt,pc,pr,datos,contenedor):
    ag=create_AG(pt,pr,pc)
    max_pop = len(pob)
    pob.sort(key=lambda  ind : ind.fi, reverse=True)
    for _ in np.arange(maxItr):
        CreateNexGen(ag,pob,datos,contenedor)
        pob = ag.Elitism(pob,max_pop)
        if pob[0].fi == 1 or (pob[0].fi-pob[-1].fi)/(pob[0].fi**2)<0.001:
            break
    return pob[0]
