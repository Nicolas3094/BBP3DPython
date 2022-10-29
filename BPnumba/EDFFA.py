from multiprocessing import reduction
from turtle import up
import numpy as np
from BPnumba.BoxN import ItemBin
from BPnumba.GeneticOperators import Hamming,MutateC1,MutateC2,MutateInversion
from BPnumba.Selection import Tournament
from BPnumba.Individual import CalcFi,Ind,createR_intidivual
from BPnumba.AGH import Crossover,FlipMutation
from BPnumba.DFFA import LightInt

from numba.typed import List as NumbaList
from numba import  njit
from typing import List

@njit
def FFSearch(mutType:int,upIndex:int , Maxitr: int, fireflyPob: List[Ind], datos: List[ItemBin], contenedor: List[int],rot:int)->Ind:
    fnum = len(fireflyPob)
    n = len(fireflyPob[0].genome)
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
                    nw = RandomMovement(mutType, fireflyPob[i])
                    if rot != 0:
                       FlipMutation(boxes=datos,gen=nw[0],rotgen=nw[1],pm=0.05,rotType=rot)
                    newFF:Ind=createR_intidivual(nw[0],nw[1])
                else:
                    nw = MoveFF(fireflyPob[i],fireflyPob[mostAttractiveFF])
                    newFF:Ind=createR_intidivual(NumbaList(nw[0]),NumbaList(nw[1]))
                    nwr = RandomMovement(mutType, newFF)
                    newFF:Ind=createR_intidivual(NumbaList(nwr[0]),NumbaList(nwr[1]))
                CalcFi(newFF, datos, contenedor,rot)
                if newFF.fi > fireflyPob[i].fi :
                    fireflyPob[i] = newFF
        fireflyPob.sort(key=lambda x:x.fi,reverse=True)
        if(fireflyPob[0].fi==1) or  (fireflyPob[0].fi-fireflyPob[-1].fi)/(fireflyPob[0].fi**2) <=0.01:
            break 
    return fireflyPob[0]

@njit
def MoveFF(firefly:Ind, ObjFirerly:Ind)->List[List[int]]:
    nwgn = Crossover(firefly,ObjFirerly)
    
    return nwgn
    
@njit
def RandomMovement(MutType,firefly:Ind)->List[List[int]]:
    if  MutType == 1:
        newgen = NumbaList(MutateC1(NumbaList(firefly.genome), NumbaList(firefly.genome_r)))
    elif MutType==2:
        newgen = NumbaList(MutateC2(NumbaList(firefly.genome),NumbaList(firefly.genome_r)))
    else:
        newgen = NumbaList(MutateInversion(NumbaList(firefly.genome), NumbaList(firefly.genome_r)))
    return newgen
  