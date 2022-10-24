import numpy as np
from numba import types, njit
from typing import List
from numba.typed import List as NumbaList
from BPnumba.Individual import Ind

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