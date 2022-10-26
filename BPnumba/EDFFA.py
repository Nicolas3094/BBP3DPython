from multiprocessing import reduction
from turtle import up
import numpy as np
from BPnumba.BoxN import ItemBin
from BPnumba.GeneticOperators import Hamming,CrossOX,Combine2,MutateC1,MutateC2,MutateInversion
from BPnumba.Selection import Tournament
from BPnumba.Individual import ind_type,create_intidivual,CalcFi,Ind,createR_intidivual
from BPnumba.AGH import Crossover,FlipMutation
from BPnumba.DFFA import LightInt

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

EDFFA_type = deferred_type()
specEF = OrderedDict()
specEF['BestInd'] = ind_type
specEF['bestfi'] = types.ListType(types.float64)
specEF['m'] = types.int64
specEF['__MutType'] = types.int64
@jitclass(specEF)
class EDFFA:
    def __init__(self,mutType:int=0):
        self.m = 0
        self.BestInd = create_intidivual(NumbaList([1]))
        self.bestfi: List[float] = NumbaList(np.zeros(1, dtype=np.float64))
        self.__MutType=mutType

    def Train(self,upIndex:int , Maxitr: int, fireflyPob: List[Ind], datos: List[ItemBin], contenedor: List[int],rot:int)->Ind:
        fnum = len(fireflyPob)
        n = len(datos)
        self.m=upIndex
        gamma=1/n
        self.bestfi: List[float] = NumbaList(np.ones(Maxitr, dtype=np.float64))
        self.BestInd = create_intidivual(NumbaList([1]))
        tmp:list[Ind] =[]
        fireflyPob.sort(key=lambda x:x.fi,reverse=True)

        for _ in np.arange(Maxitr):
            for i in np.arange(fnum):
                mostAttractiveFF:int = -1
                IAtr = -1.0
                for j in np.arange(fnum):
                    if(i==j): continue
                    dist = Hamming(fireflyPob[j], fireflyPob[i])
                    Ii = LightInt(fireflyPob[i].fi, gamma, dist)
                    Ij = LightInt(fireflyPob[j].fi, gamma,  dist)
                    if Ij > Ii and Ij>0.01:
                        if(mostAttractiveFF==-1):
                            mostAttractiveFF=j
                            IAtr=Ij
                        else:
                            if IAtr < Ij:
                                mostAttractiveFF = j
                                IAtr = Ij
                for k in np.arange(self.m):
                    newFF:Ind=create_intidivual(fireflyPob[i].genome.copy())
                    if(IAtr==-1.0): 
                        RandomMovement(self.__MutType, newFF,datos,contenedor,rot)
                    else:
                        MoveFF(newFF,fireflyPob[mostAttractiveFF])
                        if rot != 0:
                            FlipMutation(boxes=datos,gen=newFF.genome,rotgen=newFF.genome_r,pm=0.05,rotType=rot)
                        CalcFi(newFF, datos, contenedor,rot)
                    tmp.append(newFF)
            tmp.append(fireflyPob[0])
            tmp.sort(key=lambda x:x.fi,reverse=True)
            fireflyPob = tmp[:fnum]
            self.bestfi[_]=fireflyPob[0].fi
            tmp.clear()
            if(self.bestfi[_]==1):
               break 
        self.BestInd = fireflyPob[0]
        return self.BestInd
EDFFA_type.define(EDFFA.class_type.instance_type)
@njit
def createEDFFA(mutType:int=0)->EDFFA:
    return EDFFA(mutType)
@njit
def FFSearch(mutType:int,upIndex:int , Maxitr: int, fireflyPob: List[Ind], datos: List[ItemBin], contenedor: List[int],rot:int)->Ind:
    fnum = len(fireflyPob)
    n = len(datos)
    gamma=1/n
    fireflyPob.sort(key=lambda x:x.fi,reverse=True)
    
    for _ in np.arange(Maxitr):
        for i in np.arange(fnum):
            mostAttractiveFF:int = -1
            IAtr = -1.0
            for j in np.arange(fnum): #encuentra al mas atractivo
                if(i==j): continue
                dist = Hamming(fireflyPob[j], fireflyPob[i])
                Ii = LightInt(fireflyPob[i].fi, gamma, dist)
                Ij = LightInt(fireflyPob[j].fi, gamma,  dist)
                if Ij > Ii:
                    if(mostAttractiveFF==-1):
                        mostAttractiveFF=j
                        IAtr=Ij
                    else:
                        if IAtr < Ij:
                            mostAttractiveFF = j
                            IAtr = Ij
            for k in np.arange(upIndex):
                if(IAtr==-1.0): 
                    nw = RandomMovement(mutType, fireflyPob[i],datos,contenedor,rot)
                    if rot != 0:
                       FlipMutation(boxes=datos,gen=nw[0],rotgen=nw[1],pm=0.05,rotType=rot)
                    newFF:Ind=createR_intidivual(nw[0],nw[1])
                else:
                    nw = MoveFF(fireflyPob[i],fireflyPob[mostAttractiveFF])
                    newFF:Ind=createR_intidivual(NumbaList(nw[0]),NumbaList(nw[1]))
                CalcFi(newFF, datos, contenedor,rot)
                if newFF.fi > fireflyPob[i].fi:
                    fireflyPob[i] = newFF
        fireflyPob.sort(key=lambda x:x.fi,reverse=True)
        if(fireflyPob[0].fi==1):
            break 
    return fireflyPob[0]

@njit
def MoveFF(firefly:Ind, ObjFirerly:Ind)->List[List[int]]:
    nwgn = Crossover(firefly,ObjFirerly)
    return nwgn
    
@njit
def RandomMovement(MutType,firefly:Ind,DataBoxes:List[ItemBin],BinData:List[int],rot:int)->List[List[int]]:
    if  MutType == 1:
        newgen = NumbaList(MutateC1(NumbaList(firefly.genome), NumbaList(firefly.genome_r)))
    elif MutType==2:
        newgen = NumbaList(MutateC2(NumbaList(firefly.genome),NumbaList(firefly.genome_r)))
    else:
        newgen = NumbaList(MutateInversion(NumbaList(firefly.genome), NumbaList(firefly.genome_r)))
    return newgen
  