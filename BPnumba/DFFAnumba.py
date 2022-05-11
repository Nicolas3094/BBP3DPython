import numpy as np
import random
from BPnumba.NumAG import CalcFi,Hamming
from numba.typed import List as NumbaList
from numba import types, prange,njit,deferred_type
from typing import List
from collections import OrderedDict
from numba.experimental import jitclass
from BPnumba.NumAG import Ind,create_intidivual
@jitclass
class DRandMovement(object):
    def __init__(self):
        pass
    def InversionMut(self,fi_pos:List[int])->List[int]:
        n = len(fi_pos)
        step = int(n/3)
        rn = random.randint(3,n-step)
        tmp = fi_pos.copy
        tmp= tmp[0:rn]+tmp[rn:rn+step][::-1]+tmp[rn+step:n]
        return tmp
    def AlphaFF(self,fi:Ind,alpha:int)->List[List[int]]:
        newPositions = list()
        for i in np.arange(alpha):
            newPositions.append(self.InversionMut(fi.position))
        return newPositions
ranMov_type = deferred_type()
ranMov_type.define(DRandMovement.class_type.instance_type)
@njit
def instanceRanMov():
    return DRandMovement()

@njit(nogil=True)
def InstanceFFn(pob:List[List[int]],boxesData:List[List[int]], container:List[int]):
    lst = []
    for i in np.arange(len(pob)):
        ff = create_intidivual(NumbaList(pob[i]))
        CalcFi(ff,boxesData,container)
        lst.append(ff)
    return lst

@njit(nogil=True) #n2 
def BettaStep(f1:Ind,f2:Ind,gamma:float,hamming:float):
    n = len(f1.genome)
    noneNumber = n
    res = [-1 for i in np.arange(n) ]
    visited =  [False for i in np.arange(n+1)]
    for i in np.arange(n):
        if f1.genome[i] == f2.genome[i]:
            res[i] = f1.genome[i]
            noneNumber -=1 
            visited[f1.genome[i]]=True
    betta = 1/(1+gamma*(hamming**2)) #probabilidad de que tanto ''ataree'' esto, porcentaje que se van a duplicar por cada ff
    for i in np.arange(n):
        if res[i] == -1:
            pr = random.random()
            if pr < betta:
                for j in np.arange(n):
                    if not visited[f2.genome[j]]:
                        res[i]=f2.genome[j]
                        visited[f2.genome[j]]=True
                        break
            else:
                for j in np.arange(n):
                    if not visited[f1.genome[j]]:
                        res[i]=f1.genome[j]
                        visited[f1.genome[j]]=True
                        break
    return res
@njit(nogil=True)
def AlphaRandMov(firefly:Ind,alpha:int):
    movement = instanceRanMov()
    newFFs = movement.AlphaFF(firefly,alpha)
    return newFFs

@njit(nogil=True)
def AlphaStep(ff:Ind,alpha:int):
    pos = ff.genome.copy()
    n = len(pos)
    for i in np.arange(n):
        tmp = pos[i]
        pos[i] = round(pos[i] + alpha*(random.random()-0.5))
        if pos[i] < 1:
            pos[i] = 1
        elif pos[i] > n:
            pos[i]=n
        for j in np.arange(n):
            if i != j and pos[j]==pos[i]:
                pos[j] = tmp
                break
    ff.genome = NumbaList(pos)

@njit(nogil=True)
def LightInt(f1:Ind,gamma:float,dist:float):
    return f1.fi/(1+gamma*(dist**2))
@njit(nogil=True)
def RandomMov(firefly:Ind,alpha:int,DataBoxes:List[List[int]],BinData:List[int]):
    AlphaStep(firefly,alpha) # n2
    CalcFi(firefly,DataBoxes,BinData)

@njit(parallel=True)
def DFFtrain(Maxitr:int,fireflyPob:List[Ind],gamma:float,datos:List[List[int]],contenedor:List[int]):
    fnum = len(fireflyPob)
    n = len(datos[0])
    for _ in np.arange(Maxitr):
        alpha = np.floor(n-((_)/Maxitr)*(n))
        for i in np.arange(fnum-1):
            for j in prange(i+1,fnum):
                dist = Hamming(fireflyPob[j].genome,fireflyPob[0].genome)
                Ii = LightInt(fireflyPob[i],gamma,dist)
                Ij = LightInt(fireflyPob[j],gamma,dist) 
                if Ij < Ii:
                    fireflyPob[j].genome= NumbaList( BettaStep( fireflyPob[j], fireflyPob[i], gamma, dist ))
                    RandomMov(fireflyPob[j],alpha,datos,contenedor)
        RandomMov(fireflyPob[0],alpha,datos,contenedor)                   
        fireflyPob.sort(key=lambda x:x.fi,reverse=True)
    return fireflyPob[0]