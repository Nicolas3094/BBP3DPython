from multiprocessing import reduction
import numpy as np
from BPnumba.GeneticOperators import Hamming,Ind,ind_type,CalcFi,CrossOX,Combine2,MutateC1,MutateInversion
from numba.typed import List as NumbaList
from numba import  njit, deferred_type, types
from typing import List
from collections import OrderedDict
from numba.experimental import jitclass
from BPnumba.DFFA import LightInt

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

EDFFA_type = deferred_type()
specEF = OrderedDict()
specEF['BestInd'] = ind_type
specEF['bestfi'] = types.ListType(types.float64)
specEF['gamma'] = types.float64
specEF['__Heuristic'] = types.int64
specEF['__MutType'] = types.int64
@jitclass(specEF)
class EDFFA:
    def __init__(self, heuristic:int = 0,mutType:int=0):
        self.gamma = 0
        self.BestInd = Ind(NumbaList([1]))
        self.bestfi: List[float] = NumbaList(np.zeros(1, dtype=np.float64))
        self.__Heuristic= heuristic
        self.__MutType=mutType

    def Train(self, Maxitr: int, fireflyPob: List[Ind], datos: List[List[int]], contenedor: List[int])->Ind:
        fnum = len(fireflyPob)
        n = len(datos)
        if self.gamma==0:
            self.gamma=1/n
        rd: List[float] = []
        self.bestfi: List[float] = NumbaList(np.zeros(1, dtype=np.float64))
        bestFF=len(fireflyPob)-1
        self.BestInd = Ind(NumbaList([1]))
        fireflyPob.sort(key=lambda x: x.fi)

        for _ in np.arange(Maxitr):
            alpha = np.floor(n-((_+1)/Maxitr)*(n))
            for i in np.arange(fnum-1):
                for j in np.arange(i+1,fnum):
                    dist = Hamming(NumbaList(fireflyPob[j].genome), NumbaList(fireflyPob[i].genome))
                    Ii = LightInt(fireflyPob[i].fi, self.gamma, dist)
                    Ij = LightInt(fireflyPob[j].fi, self.gamma,  dist)
                    if Ij > Ii:
                        self.MoveFF(fireflyPob[i], fireflyPob[j], dist)
                        CalcFi(fireflyPob[i], datos, contenedor,self.__Heuristic)
            rd.append(fireflyPob[bestFF].fi)
            if fireflyPob[bestFF].fi == 1:
                break
            self.RandomMove(fireflyPob[bestFF], alpha, datos, contenedor)
            fireflyPob.sort(key=lambda x: x.fi)
        self.BestInd = fireflyPob[bestFF]
        rd = np.array(rd, dtype=np.float64)
        self.bestfi = NumbaList(rd)
        return self.BestInd
    def MoveFF(self, firefly:Ind, ObjFirerly:Ind,dist)->None:
        betta = int(len(firefly.genome)*(1/(1+self.gamma*dist*dist))) #pasos que dependen de dist y gamma
        nwgn = NumbaList(EBettaStep(firefly.genome, ObjFirerly.genome, betta))
        firefly.genome = nwgn

    def RandomMove(self, firefly:Ind,alpha:int,DataBoxes:List[List[int]],BinData:List[int])->None:
        if  self.__MutType == 1:
            newgen = NumbaList(MutateC1(firefly.genome, alpha))
        elif self.__MutType==2:
            newgen = NumbaList(EAlphaStepC2(firefly.genome, alpha))
        else:
            newgen = NumbaList(MutateInversion(firefly.genome, alpha))
        firefly.genome = newgen 
        CalcFi(firefly, DataBoxes, BinData,self.__Heuristic)

    def SelectHeuristic(self, hID:int)->None: #solo hay 2,0 para el DBLF, DBLF2 en cualquier otro caso
        self.__Heuristic= hID

EDFFA_type.define(EDFFA.class_type.instance_type)
@njit
def createEDFFA(heuristic:int=0,mutType:int=0)->EDFFA:
    return EDFFA(heuristic,mutType)