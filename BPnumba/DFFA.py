import numpy as np
import random
from BPnumba.GeneticOperators import Hamming,SwapPointValue
from BPnumba.NumAG import CalcFi

from numba.typed import List as NumbaList
from numba import  njit, deferred_type, types
from typing import List
from collections import OrderedDict
from numba.experimental import jitclass
from BPnumba.GeneticOperators import Ind, ind_type

@njit
def BettaStep(f1: List[int], f2:  List[int], betta: float):
    n = len(f1)
    for i in np.arange(n):
        if f1[i] == f2[i]:
            continue
        pr = random.random()
        if pr < betta:
            SwapPointValue(f1,i,f2[i])
@njit
def AlphaStep(genome: List[int], alpha: int):
    n = len(genome)
    minL = 1
    maxL = n
    for i in np.arange(n):
        tmp = genome[i]
        posVal = abs(round(genome[i] + alpha*(random.random()-0.5)))
        if posVal < minL:
            posVal=minL
            minL +=1
        elif genome[i] > maxL:
            posVal=maxL
            maxL -=1
        elif genome[i] == tmp:
            continue
        SwapPointValue(genome,i,posVal)

@njit
def LightInt(f1: float, gamma: float, dist: float):
    return f1/(1+gamma*(dist**2))


DFFA_type = deferred_type()
specF = OrderedDict()
specF['BestInd'] = ind_type
specF['bestfi'] = types.ListType(types.float64)
specF['gamma'] = types.float64
specF['__Heuristic'] = types.int64
@jitclass(specF)
class DFFA:
    def __init__(self, gamma: float,heuristic:int = 0):
        self.gamma = gamma
        self.BestInd = Ind(NumbaList([1]))
        self.bestfi: List[float] = NumbaList(np.zeros(1, dtype=np.float64))
        self.__Heuristic= heuristic
    def Train(self, Maxitr: int, fireflyPob: List[Ind], datos: List[List[int]], contenedor: List[int]):
        fnum = len(fireflyPob)
        n = len(datos)
        self.bestfi: List[float] = NumbaList(np.zeros(Maxitr, dtype=np.float64))
        fireflyPob.sort(key=lambda x: x.fi)
        actualItr = 0
        for _ in np.arange(Maxitr):
            alpha = np.floor(n-((_)/Maxitr)*(n))
            for i in np.arange(fnum-1):
                for j in np.arange(i+1, fnum):
                    dist = Hamming(NumbaList(fireflyPob[j].genome), NumbaList(
                        fireflyPob[i].genome))
                    Ii = LightInt(fireflyPob[i].fi, self.gamma, dist)
                    Ij = LightInt(fireflyPob[j].fi, self.gamma,  dist)
                    if Ii < Ij:
                        self.MoveFF(fireflyPob[i], fireflyPob[j], dist)
                        self.RandomMove(fireflyPob[i], alpha, datos, contenedor)
                    elif dist == 0:
                        self.RandomMove(fireflyPob[i], alpha, datos, contenedor)
            self.bestfi[_] = fireflyPob[fnum-1].fi
            actualItr =_
            if fireflyPob[fnum-1].fi == 1:
                self.BestInd = fireflyPob[fnum-1]
                break
            self.RandomMove(fireflyPob[fnum-1], alpha, datos, contenedor)    
            fireflyPob.sort(key=lambda x: x.fi)
        self.BestInd = fireflyPob[fnum-1]
        if actualItr != Maxitr:
            self.bestfi =self.bestfi[:actualItr]
        return self.BestInd
    def MoveFF(self, firefly:Ind, ObjFirerly:Ind,dist):
        betta: float = 1/(1+self.gamma*dist*dist)
        BettaStep(firefly.genome, ObjFirerly.genome, betta)
    def RandomMove(self, firefly:Ind,alpha:int,DataBoxes:List[List[int]],BinData:List[int]):
        AlphaStep(firefly.genome, alpha)  # n2
        CalcFi(firefly, DataBoxes, BinData,self.__Heuristic)

DFFA_type.define(DFFA.class_type.instance_type)


@njit
def createDFFA(gamma: float = 0,heuristic:int=0):
    return DFFA(gamma,heuristic)
