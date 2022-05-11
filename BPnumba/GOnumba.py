import numpy as np
import random
from numba.typed import List as NumbaList
from numba import types, prange,njit,deferred_type,typed
from typing import List
from numba.experimental import jitclass
from numba.typed import List as NumbaList

@njit
def CreatePermutation(ls1:List[int]):
    repeated = []
    while len(repeated)!=len(ls1):
        randNum = random.randint(1,len(ls1))
        if randNum not in repeated:
            repeated.append(randNum)
    return repeated
@njit
def CreatePoblation(num:int, ls2:List[int])->List[List[int]]:
    poblation = []
    for _ in np.arange(num):
        pi =  CreatePermutation(ls2)
        if pi not in poblation:
            poblation.append(pi)
    return poblation


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

@njit(nogill=True)
def InverseMutation(gen:List[int],i,j)->None:
    tmp =gen.copy()
    for k in np.arange(i,j+1):
        gen[k] = tmp[j-k+i]

@njit(nogil=True)
def Hamming(f1:List[int],f2:List[int])->float:
    count =0
    for i in np.arange(len(f1)):
        if f1[i] !=f2[i]:
            count +=1
    return np.float64(count)