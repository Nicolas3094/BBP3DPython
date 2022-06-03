from dataclasses import replace
import numpy as np
import random
from BPnumba.GeneticOperators import Hamming,SwapPointValue,Ind, ind_type,CalcFi,InverseMutation
from numba.typed import List as NumbaList
from numba import  njit, deferred_type, types
from typing import List
from collections import OrderedDict
from numba.experimental import jitclass

@njit
def BettaStep(f1: List[int], f2:  List[int], betta: float)->None:
    n = len(f1)
    for i in np.arange(n):
        if f1[i] == f2[i]:
            continue
        pr = np.random.random()
        if pr < betta:
            SwapPointValue(f1,i,f2[i])
@njit
def AlphaStep(genome: List[int], alpha: int)->None:
    n = len(genome)
    minL = 1
    maxL = n
    for i in np.arange(n):
        tmp = genome[i]
        posVal = abs(round(genome[i] + alpha*(np.random.random()-0.5)))
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
def LightInt(f1: float, gamma: float, dist: float)->float:
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
    def Train(self, Maxitr: int, fireflyPob: List[Ind], datos: List[List[int]], contenedor: List[int])->Ind:
        fnum = len(fireflyPob)
        n = len(datos)
        rd: List[float] = []
        self.bestfi: List[float] = NumbaList(np.zeros(1, dtype=np.float64))
        fireflyPob.sort(key=lambda x: x.fi)
        for _ in np.arange(Maxitr):
            alpha = np.floor(n-((_+1)/Maxitr)*(n))
            for i in np.arange(fnum-1):
                for j in np.arange(i+1, fnum):
                    dist = Hamming(NumbaList(fireflyPob[j].genome), NumbaList(fireflyPob[i].genome))
                    Ii = LightInt(fireflyPob[i].fi, self.gamma, dist)
                    Ij = LightInt(fireflyPob[j].fi, self.gamma,  dist)
                    if Ii < Ij:
                        self.MoveFF(fireflyPob[i], fireflyPob[j], dist)
                        self.RandomMove(fireflyPob[i], alpha, datos, contenedor)
            rd.append(fireflyPob[fnum-1].fi)
            if fireflyPob[fnum-1].fi == 1:
                self.BestInd = fireflyPob[fnum-1]
                break
            self.RandomMove(fireflyPob[fnum-1], alpha, datos, contenedor)    
            fireflyPob.sort(key=lambda x: x.fi)
        self.BestInd = fireflyPob[fnum-1]
        rd = np.array(rd, dtype=np.float64)
        self.bestfi = NumbaList(rd)
        return self.BestInd
    def MoveFF(self, firefly:Ind, ObjFirerly:Ind,dist:float)->None:
        betta: float = 1/(1+self.gamma*dist*dist)
        BettaStep(firefly.genome, ObjFirerly.genome, betta)
    def RandomMove(self, firefly:Ind,alpha:int,DataBoxes:List[List[int]],BinData:List[int])->None:
        AlphaStep(firefly.genome, alpha)  # n2
        CalcFi(firefly, DataBoxes, BinData,self.__Heuristic)
    def SelectHeuristic(self, hID:int)->None:
        self.__Heuristic= hID
DFFA_type.define(DFFA.class_type.instance_type)


@njit
def createDFFA(gamma: float = 0,heuristic:int=0)->DFFA:
    return DFFA(gamma,heuristic)


@njit
def DBettaStep(f1: List[int], f2:  List[int], betta: float)->None:
    n = len(f1)
    for i in np.arange(n):
        if f1[i] == f2[i]:
            continue
        pr = np.random.random()
        if pr < betta:
            SwapPointValue(f1,i,f2[i])
@njit
def DAlphaStep(genome: List[int], alpha: int)->None:
    n = len(genome)
    indexV = NumbaList(np.asarray(np.random.choice(n,int(alpha), replace=False),dtype=np.int64))
    valuesV = NumbaList(np.asarray(np.random.choice(n,int(alpha),replace=False)+1,dtype=np.int64))
    for i in np.arange(int(alpha),dtype=np.int64):
        SwapPointValue(genome,indexV[i],valuesV[i])

EDFFA_type = deferred_type()
specEF = OrderedDict()
specEF['BestInd'] = ind_type
specEF['bestfi'] = types.ListType(types.float64)
specEF['gamma'] = types.float64
specEF['__Heuristic'] = types.int64
@jitclass(specEF)
class EDFFA:
    def __init__(self, gamma: float,heuristic:int = 0):
        self.gamma = gamma
        self.BestInd = Ind(NumbaList([1]))
        self.bestfi: List[float] = NumbaList(np.zeros(1, dtype=np.float64))
        self.__Heuristic= heuristic
    def Train(self, Maxitr: int, fireflyPob: List[Ind], datos: List[List[int]], contenedor: List[int])->Ind:
        fnum = len(fireflyPob)
        n = len(datos)
        rd: List[float] = []
        self.bestfi: List[float] = NumbaList(np.zeros(1, dtype=np.float64))
        fireflyPob.sort(key=lambda x: x.fi)
        bestFF=fnum-1
        for _ in np.arange(Maxitr):
            alpha = np.floor(n-((_+1)/Maxitr)*(n))
            for i in np.arange(fnum):
                if bestFF ==i:
                    continue
                dist = Hamming(NumbaList(fireflyPob[bestFF].genome), NumbaList(fireflyPob[i].genome))
                Ii = LightInt(fireflyPob[i].fi, self.gamma, dist)
                Ij = LightInt(fireflyPob[bestFF].fi, self.gamma,  dist)
                if Ij > Ii:
                    self.MoveFF(fireflyPob[i], fireflyPob[bestFF], dist)
                    self.RandomMove(fireflyPob[i], alpha, datos, contenedor)
                    if fireflyPob[i].fi > fireflyPob[bestFF].fi:
                        bestFF = i
                else:
                    self.RandomMove(fireflyPob[i], alpha, datos, contenedor)
                    if fireflyPob[i].fi > fireflyPob[bestFF].fi:
                        bestFF = i
            rd.append(fireflyPob[bestFF].fi)
            if fireflyPob[bestFF].fi == 1:
                break
        self.BestInd = fireflyPob[bestFF]
        rd = np.array(rd, dtype=np.float64)
        self.bestfi = NumbaList(rd)
        return self.BestInd
    def MoveFF(self, firefly:Ind, ObjFirerly:Ind,dist)->None:
        betta: float = 1/(1+self.gamma*dist*dist)
        DBettaStep(firefly.genome, ObjFirerly.genome, betta)
    def RandomMove(self, firefly:Ind,alpha:int,DataBoxes:List[List[int]],BinData:List[int])->None:
        DAlphaStep(firefly.genome, alpha)  # n2
        CalcFi(firefly, DataBoxes, BinData,self.__Heuristic)
    def SelectHeuristic(self, hID:int)->None:
        self.__Heuristic= hID
EDFFA_type.define(EDFFA.class_type.instance_type)


@njit
def createEDFFA(gamma: float = 0,heuristic:int=0)->EDFFA:
    return EDFFA(gamma,heuristic)