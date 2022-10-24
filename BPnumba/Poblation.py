import numpy as np
from numba import  njit,prange
from typing import List
from numba.typed import List as NumbaList
from BPnumba.BoxN import ItemBin
from BPnumba.Individual import Ind,create_intidivual,createR_intidivual,CalcFi
@njit
def CreatePermutation(ls1:int)->List[int]:
    return NumbaList(np.random.choice(ls1,ls1, replace=False)+1)
@njit
def SortPoblationCode(Data:List[ItemBin],n:int,reverse:bool=True)->List[List[int]]:
    sortedPob:list[list[int]]= []
    
    Data.sort(key=lambda bx : bx.RDim()[0]*bx.RDim()[1]*bx.RDim()[2],reverse=reverse)
    nwcd = np.zeros(n,dtype=np.int64)
    for k in np.arange(n):
        nwcd[k] = Data[k].id
    sortedPob.append(NumbaList(nwcd))
    
    Data.sort(key=lambda bx : bx.RDim()[0],reverse=reverse)
    nwcd = np.zeros(n,dtype=np.int64)
    for k in np.arange(n):
        nwcd[k] = Data[k].id
    sortedPob.append(NumbaList(nwcd))

    Data.sort(key=lambda bx : bx.RDim()[1],reverse=reverse)
    nwcd = np.zeros(n,dtype=np.int64)
    for k in np.arange(n):
        nwcd[k] = Data[k].id
    sortedPob.append(NumbaList(nwcd))

    Data.sort(key=lambda bx : bx.RDim()[2],reverse=reverse)
    nwcd = np.zeros(n,dtype=np.int64)
    for k in np.arange(n):
        nwcd[k] = Data[k].id
    sortedPob.append(NumbaList(nwcd))

    return NumbaList(sortedPob)
@njit 
def CreateHeuristicPob(num:int,Data:List[ItemBin],reversed=True)->List[List[int]]:
    poblation:list[list[int]] = SortPoblationCode(Data=Data,n=len(Data),reverse=reversed)
    k = len(poblation)
    for _ in np.arange(num-k):
        poblation.append(CreatePermutation(len(Data)))

    return poblation

@njit
def CreatePoblation(num:int, ls2:List[int])->List[List[int]]:
    poblation = np.zeros(shape=(num, len(ls2)), dtype=np.int64)
    for _ in np.arange(num):
        poblation[_] = CreatePermutation(ls2)
    return NumbaList(poblation)

@njit
def CreateRotHPob(pop :int, ITEMS:list[ItemBin], rotType:int=0)->list[Ind]:
    hpob = CreateHeuristicPob(num = pop,Data = ITEMS)
    indPob:list[Ind] = []
    ls = np.zeros(len(ITEMS),dtype=np.int64)
    if rotType == 2:
        for ind in hpob:
            for i in  np.arange(len(ITEMS)):
                ls[i] = np.random.randint(2)
            indPob.append(createR_intidivual(ind,NumbaList(ls)))
    elif rotType== 6:
         for ind in hpob:
            for i in  np.arange(len(ITEMS)):
                ls[i] = np.random.randint(6)
            indPob.append(createR_intidivual(ind,NumbaList(ls)))
    else:
        for ind in hpob:
            indPob.append(create_intidivual(ind))
    return NumbaList(indPob)
@njit(parallel=True)
def EvalPob(pob:List[Ind],boxData:List[ItemBin], container :List[int],rotation:int=0):
    for i in prange(len(pob)):
        CalcFi(ind = pob[i], boxesData=boxData,container=container,rotation=rotation)