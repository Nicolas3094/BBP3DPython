
import numpy as np
from numba import njit,objmode, prange
from typing import List
from numba.typed import List as NumbaList
import time
from BPnumba.GeneticOperators import Ind, CreateHeuristicPob, InstancePob,CodeSolution

@njit
def ReduceDim(Data:List[List[int]],bin:List[int])->List[List[int]]:
    convertedData:List[List[int]]= []
    mp = dict()
    Ub = len(Data)
    for i in np.arange(Ub):
        mp[CodeSolution(Data[i])]=False
    for i in np.arange(Ub):
        if not mp[CodeSolution(Data[i])]:
            mp[CodeSolution(Data[i])]=True
            convertedData.append(list(Data[i]))
        else:
            for vec in convertedData:
                if vec[0] == Data[i][0] and vec[1] == Data[i][1] and vec[2] == Data[i][2]:
                    if 2*vec[2] <= bin[2]:
                        vec[2] *= 2
                    elif 2*vec[1] <= bin[1]:
                        vec[1] *= 2
                    elif 2*vec[0] <= bin[0]:
                        vec[0] *= 2
                    else:
                        convertedData.append(list(Data[i]))
                    break
    return convertedData 

@njit(parallel=True)
def Test(boxes:List[List[List[int]]],bin:List[int],alg, maxItr: int, lst: List[List[float]],heuristic:int=0):
    DATA = boxes.copy()
    for i in prange(20):
        DAT:List[List[int]]= NumbaList(DATA[i])
        n:int=len(DAT)
        Genomes:List[int] =  CreateHeuristicPob(50,DAT,bin)
        Pob:List[Ind]= InstancePob(Genomes,DAT, bin, heuristic)
        with objmode(time1='f8'):
            time1 = time.perf_counter()
        bestInd:Ind = alg.Train(maxItr, Pob, DAT, bin)
        with objmode(last='f8'):
            last = time.perf_counter() - time1      
        epochs = len(alg.bestfi)
        if heuristic == 0:
            lst[i] = np.array([bestInd.fi, n-bestInd.load, epochs, last],dtype=np.float64)
        else:
            lst[20+i] = np.array([bestInd.fi, n-bestInd.load,epochs, last], dtype=np.float64) 
    