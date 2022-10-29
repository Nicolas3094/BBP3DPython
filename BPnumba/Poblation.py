import numpy as np
from numba import  njit,prange,objmode
from typing import List
from numba.typed import List as NumbaList
from BPnumba.BoxN import ItemBin
from BPnumba.Individual import Ind,create_intidivual,createR_intidivual,CalcFi
import time

@njit
def CreatePermutation(UBOXES:List[ItemBin]):
    n=len(UBOXES)
    return NumbaList(np.random.choice(n, n, replace=False)+1)
@njit
def TotalItems(UBOXES:List[ItemBin]):
    n=len(UBOXES)
    num=0
    for i in np.arange(n):
        num += UBOXES[i].n
    return num
@njit
def MakeReapeted(n:int,types_num:int,BoxData:List[ItemBin]):
    nwcd = np.zeros(n,dtype=np.int64)
    acc=0
    for t in np.arange(types_num):
        nw =np.zeros(BoxData[t].n,dtype=np.int64)+BoxData[t].id
        for j in np.arange(BoxData[t].n):
            nwcd[acc+j]=nw[j]
        acc += BoxData[t].n
    return NumbaList(nwcd)

@njit
def MakeGen(n:int,BoxData:List[ItemBin])->List[int]:
    nwcd = np.zeros(n,dtype=np.int64)
    for i in np.arange(n):
        nwcd[i]=BoxData[i].id
    return NumbaList(nwcd)

@njit
def SortPoblationCode(Data:List[ItemBin],reverse:bool=True)->List[List[int]]:
    
    sortedPob:list[list[int]]= []
    
    n = len(Data)
    
    Original = Data.copy()
    
    Data.sort(key=lambda bx : bx.RDim()[0]*bx.RDim()[1]*bx.RDim()[2],reverse=reverse)
    
    sortedPob.append(MakeGen(n,Data))
    
    Data.sort(key=lambda bx : bx.RDim()[0],reverse=reverse)

    sortedPob.append(MakeGen(n,Data))

    Data.sort(key=lambda bx : bx.RDim()[1],reverse=reverse)

    sortedPob.append(MakeGen(n,Data))

    Data.sort(key=lambda bx : bx.RDim()[2],reverse=reverse)

    sortedPob.append(MakeGen(n,Data))

    Data=Original.copy()
    
    return NumbaList(sortedPob)
@njit 
def CreateHeuristicPob(num:int,Data:List[ItemBin],reversed=True)->List[List[int]]:
    poblation:list[list[int]] = SortPoblationCode(Data=Data,reverse=reversed)
    k = len(poblation)
    for _ in np.arange(num-k):
        poblation.append(CreatePermutation(Data))
    return poblation


@njit
def CreateRotHPob(pop :int, ITEMS:list[ItemBin], rotType:int=0)->list[Ind]:
    hpob = CreateHeuristicPob(pop,ITEMS)
    indPob:list[Ind] = []
    n = len(ITEMS)
    ls = np.zeros(n,dtype=np.int64)
    if rotType != 0:
        for ind in hpob:
            for i in  np.arange(n): 
                ls[i] = np.random.randint(rotType)
                box = ITEMS[ind[i]-1]
                box.rotate(ls[i],rotType)
                while not box.isValidRot():
                    ls[i] = np.random.randint(rotType)
                    box.rotate(ls[i],rotType)
                box.rotate(0,rotType)
            indPob.append(createR_intidivual(ind,NumbaList(ls)))
    else:
        for ind in hpob:
            indPob.append(create_intidivual(ind))
    return NumbaList(indPob)
@njit
def EvalPob(pob:List[Ind],boxData:List[ItemBin], container :List[int],rotation:int=0):
    for i in np.arange(len(pob)):
        CalcFi(ind = pob[i], boxesData=boxData,container=container,rotation=rotation)
