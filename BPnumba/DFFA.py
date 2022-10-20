import numpy as np
from BPnumba.GeneticOperators import Hamming, SwapPointValue,CrossOX,Combine2,MutateC1,MutateInversion
from BPnumba.Individual import Ind,ind_type,CalcFi
from numba.typed import List as NumbaList
from numba import  njit, deferred_type, types
from typing import List
from collections import OrderedDict
from numba.experimental import jitclass

@njit
def EBettaStep(f1: List[int], f2:  List[int], betta: int)->List[int]:
    n = len(f1)
    init = np.random.randint(0,n-2) #Existe la posibilidad que el punto inicial casi al final del genoma
    if init + betta > n-1:
        end=int(n-1)
    else:
        end=int(init+betta)
    return CrossOX(f2,f1,init,end)
    
@njit
def EAlphaStepC2(genome: List[int], alpha: int)->List[int]:
    n = int(len(genome))
    init = np.random.randint(1,int(n/2))
    if init + alpha > n-2:
        end = n-2
    else:
        end =  np.random.randint(init+1,n-1)
    if np.random.random()<0.5:
        index=np.random.randint(0,init)
    else:
        index=np.random.randint(end+1,n)
    newcode = Combine2(genome,index,init,end)
    return newcode
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
        if posVal > maxL:
            posVal -= maxL
        if posVal < minL:
            posVal=minL
        if posVal == maxL:
            maxL-=1
        if posVal == minL:
            minL+=1
        if posVal == tmp:
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
specF['__MutType'] = types.int64

@jitclass(specF)
class DFFA:
    def __init__(self, heuristic:int,mutType:int):
        self.gamma = 0
        self.BestInd = Ind(NumbaList([1]))
        self.bestfi: List[float] = NumbaList(np.zeros(1, dtype=np.float64))
        self.__Heuristic= heuristic
        self.__MutType=mutType

    def Train(self, Maxitr: int, fireflyPob: List[Ind], datos: List[List[int]], contenedor: List[int])->Ind:
        fnum = len(fireflyPob)
        n = len(datos)
        self.bestfi: List[float] = NumbaList(np.ones(Maxitr, dtype=np.float64))
        self.gamma=1/n
        fireflyPob.sort(key=lambda x: x.fi)
        self.BestInd = fireflyPob[-1]
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
            fireflyPob.sort(key=lambda x: x.fi)
            if(self.BestInd.fi<fireflyPob[-1].fi):
                self.BestInd = fireflyPob[-1]
            self.bestfi[_] =fireflyPob[-1].fi
            if fireflyPob[-1].fi == 1:
                break
            #self.RandomMove(fireflyPob[-1], alpha, datos, contenedor)    
        self.BestInd = fireflyPob[fnum-1]
        return self.BestInd

    def MoveFF(self, firefly:Ind, ObjFirerly:Ind,dist:float)->None:
        betta: float = 1/(1+self.gamma*dist*dist)
        BettaStep(firefly.genome, ObjFirerly.genome, betta)

    def RandomMove(self, firefly:Ind,alpha:int,DataBoxes:List[List[int]],BinData:List[int])->None:
        if  self.__MutType == 1:
            newgen = NumbaList(MutateC1(firefly.genome, alpha))
            firefly.genome = newgen 
        elif self.__MutType==2:
            newgen = NumbaList(EAlphaStepC2(firefly.genome, alpha))
            firefly.genome = newgen 
        elif self.__MutType==3:
            newgen = NumbaList(MutateInversion(firefly.genome, alpha))
            firefly.genome = newgen 
        else:
            AlphaStep(firefly.genome, alpha)  # n2

        CalcFi(firefly, DataBoxes, BinData,self.__Heuristic)

    def SelectHeuristic(self, hID:int)->None:
        self.__Heuristic= hID

DFFA_type.define(DFFA.class_type.instance_type)
@njit
def createDFFA(heuristic:int=0,mutType:int=0)->DFFA:
    return DFFA(heuristic,mutType)