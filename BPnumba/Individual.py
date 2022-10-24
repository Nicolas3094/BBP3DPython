import numpy as np
import random
from numba import types, njit,deferred_type
from typing import List
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from collections import OrderedDict
from BPnumba.BoxN import ItemBin
from BPnumba.BPPH import create_Bin,DBLF


sepecInd = OrderedDict()
sepecInd['fi'] = types.float64
sepecInd['genome'] = types.ListType(types.int64)
sepecInd['genome_r'] = types.ListType(types.int64)
sepecInd['load'] = types.int64
sepecInd['codeSolution'] =  types.string
@jitclass(sepecInd)
class Ind:
    def __init__(self,genome:List[int],rgenome:List[int]):
         self.fi:float = 0
         self.genome = genome
         self.genome_r = rgenome
         self.load = 0
         self.codeSolution = ''
ind_type = deferred_type()
ind_type.define(Ind.class_type.instance_type)

@njit
def create_intidivual(gen:List[int])->Ind:
    return Ind(gen,NumbaList(np.zeros(len(gen),dtype=np.int64)))

@njit
def createR_intidivual(gen:List[int],rotgen : List[int])->Ind:
    return Ind(gen,NumbaList(rotgen))
@njit 
def CodeSolution(idLoaded:List[int])->types.unicode_type:
    st = "|"
    for i in np.arange(len(idLoaded)):
        st += str(idLoaded[i])+"|"
    return st

@njit
def CalcFi(ind:Ind, boxesData:List[ItemBin], container:List[int],rotation:int=0):

    bin = create_Bin(NumbaList(container)) # Insrancio Contenedor

    DBLF(
        bin = bin, 
        itemsToPack= ind.genome,
        itemsRor= ind.genome_r,
        BoxesData = boxesData.copy(),
        wayRotation=rotation
        ) #Implementa heuristica
        
    ind.load = bin.getN() #Numero de cajas dentro
    
    ind.fi =  bin.getLoadVol()/(container[0]*container[1]*container[2]) #Fumncion aptitud
    
    ind.codeSolution = CodeSolution(bin.getBoxes()) #Solucion

@njit
def EvalPoblation(pob:List[Ind], boxesData:List[ItemBin], container:List[int],rotation:int=0):
    pop = len(pob)
    for i in np.arange(pop):
        CalcFi(pop[i],boxesData,container,rotation)

@njit
def InstancePob(pob:List[List[int]],boxesData:List[List[int]], container:List[int],rotation:int=0)->List[Ind]:
    lst = [ ]
    for i in np.arange(len(pob)):
        ind:Ind = create_intidivual(NumbaList(pob[i]))
        CalcFi(ind,boxesData,container,rotation)
        lst.append(ind)
    return lst
