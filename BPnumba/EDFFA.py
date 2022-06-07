import numpy as np
from BPnumba.GeneticOperators import Hamming,SwapPointValue,Ind, create_intidivual, ind_type,CalcFi,InverseMutation,CrossOX
from numba.typed import List as NumbaList
from numba import  njit, deferred_type, types
from typing import List
from collections import OrderedDict
from numba.experimental import jitclass
from BPnumba.DFFA import LightInt

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
def EAlphaStep(genome: List[int], alpha: int)->List[int]:
    n = len(genome)
    init = np.random.randint(1,n-2)
    if init+alpha>n-1:
        end=n-1
    else:
        end=init+alpha
    return InverseMutation(genome,init,end)

EDFFA_type = deferred_type()
specEF = OrderedDict()
specEF['BestInd'] = ind_type
specEF['bestfi'] = types.ListType(types.float64)
specEF['gamma'] = types.float64
specEF['__Heuristic'] = types.int64
@jitclass(specEF)
class EDFFA:
    def __init__(self, gamma: float=0,heuristic:int = 0):
        self.gamma = gamma
        self.BestInd = Ind(NumbaList([1]))
        self.bestfi: List[float] = NumbaList(np.zeros(1, dtype=np.float64))
        self.__Heuristic= heuristic
    def Train(self, Maxitr: int, fireflyPob: List[Ind], datos: List[List[int]], contenedor: List[int])->Ind:
        fnum = len(fireflyPob)
        n = len(datos)
        if self.gamma==0:
            self.gamma=1/n
        rd: List[float] = []
        self.bestfi: List[float] = NumbaList(np.zeros(1, dtype=np.float64))
        bestFF=0
        self.BestInd = Ind(NumbaList([1]))
        for i in np.arange(fnum):
            if fireflyPob[i].fi > fireflyPob[bestFF].fi and bestFF != i:
                bestFF=i
        for _ in np.arange(Maxitr):
            alpha = np.floor(n-((_+1)/Maxitr)*(n))
            prev = bestFF
            for i in np.arange(fnum):
                if bestFF ==i:
                    continue
                dist = Hamming(NumbaList(fireflyPob[bestFF].genome), NumbaList(fireflyPob[i].genome))
                Ii = LightInt(fireflyPob[i].fi, self.gamma, dist)
                Ij = LightInt(fireflyPob[bestFF].fi, self.gamma,  dist)
                if Ij > Ii:
                    self.MoveFF(fireflyPob[i], fireflyPob[bestFF], dist)
                    CalcFi(fireflyPob[i], datos, contenedor,self.__Heuristic)
                    #self.RandomMove(fireflyPob[i], alpha, datos, contenedor)
                    if fireflyPob[i].fi > fireflyPob[bestFF].fi:
                        bestFF = i
                else:
                    self.RandomMove(fireflyPob[i], alpha, datos, contenedor)
                    if fireflyPob[i].fi > fireflyPob[bestFF].fi:
                        bestFF = i
            rd.append(fireflyPob[bestFF].fi)
            if fireflyPob[bestFF].fi == 1:
                break
            if prev == bestFF:
                self.RandomMove(fireflyPob[bestFF],alpha,datos,contenedor)
                for i in np.arange(fnum):
                    if fireflyPob[i].fi > fireflyPob[bestFF].fi and bestFF != i:
                        bestFF=i
        self.BestInd = fireflyPob[bestFF]
        rd = np.array(rd, dtype=np.float64)
        self.bestfi = NumbaList(rd)
        return self.BestInd
    def MoveFF(self, firefly:Ind, ObjFirerly:Ind,dist)->None:
        if dist <=1:
            return
        betta = int(len(firefly.genome)*(1/(1+self.gamma*dist*dist))) #pasos que dependen de dist y gamma
        nwgn = NumbaList(EBettaStep(firefly.genome, ObjFirerly.genome, betta))
        firefly.genome = nwgn

    def RandomMove(self, firefly:Ind,alpha:int,DataBoxes:List[List[int]],BinData:List[int])->None:
        if alpha == 0:
            return
        newgen = NumbaList(EAlphaStep(firefly.genome, alpha))
        firefly.genome = newgen 
        CalcFi(firefly, DataBoxes, BinData,self.__Heuristic)

    def SelectHeuristic(self, hID:int)->None: #solo hay 2,0 para el DBLF, DBLF2 en cualquier otro caso
        self.__Heuristic= hID

EDFFA_type.define(EDFFA.class_type.instance_type)
@njit
def createEDFFA(gamma: float = 0,heuristic:int=0)->EDFFA:
    return EDFFA(gamma,heuristic)