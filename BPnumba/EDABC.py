import numpy as np
from numba.typed import List as NumbaList

from BPnumba.GeneticOperators import MutateC2
from BPnumba.Poblation import CreatePermutation
from BPnumba.Selection import Tournament
from BPnumba.Individual import ind_type,create_intidivual,CalcFi,Ind

from numba import njit, deferred_type, types
from typing import List
import random
from numba.experimental import jitclass
from collections import OrderedDict

EABC_type = deferred_type()
specEABC = OrderedDict()
specEABC['pop_num'] = types.int64
specEABC['n'] = types.int64
specEABC['BestInd'] = ind_type
specEABC['Limit'] = types.int64
specEABC['fail'] = types.ListType(types.ListType(types.int64))
specEABC['bestfi'] = types.ListType(types.float64)
specEABC['__Heuristic'] = types.int64
@jitclass(specEABC)
class EDABC:
    def __init__(self, heuristic:int=0):
        self.pop_num = 0
        self.n = 0 
        self.BestInd = Ind(NumbaList([1]))
        self.Limit = 0
        listaL = [ NumbaList([i,0]) for i in np.arange(1)]
        self.fail:List[List[int]] = NumbaList(listaL)
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64)) 
        self.__Heuristic = heuristic

    def Train(self,numItr: int, ColonyWorker: List[Ind], datos:List[List[int]], contenedor:List[int])->Ind:
        
        self.pop_num = len(ColonyWorker)
        self.n = len(datos)
        listaL = [ NumbaList([i,0]) for i in np.arange(self.pop_num)]
        self.fail:List[List[int]] = NumbaList(listaL)
        self.bestfi:List[float] = NumbaList(np.ones(numItr,dtype=np.float64))
        self.BestInd = Ind(NumbaList([1]))
        self.Limit = round(np.sqrt(self.pop_num*self.n))
        
        for bee in ColonyWorker:
            if bee.fi > self.BestInd.fi:
                self.BestInd=bee
        
        for _ in np.arange(numItr):
            # Busqueda local de abejas para todas las trabajadoras y calcila Pi
            self.WorkerBeePhase(ColonyWorker,datos,contenedor)

            # Busqueda Onlooker bee
            self.OnlookerPhase(ColonyWorker,datos,contenedor)

            #Busqueda de mejoras de acuerdo al Limite
            self.ScoutPhase(ColonyWorker,datos,contenedor)

            self.bestfi[_]=self.BestInd.fi

            if self.BestInd.fi == 1:
                break
            
        return self.BestInd

    def ImproveFlower(self,i:int,ColonyWorker:List[Ind],datos:List[List[int]],contenedor:List[int])->None:
        nwGen = self.ImproveB(i, ColonyWorker)
        newBee = create_intidivual(NumbaList(nwGen))
        CalcFi(newBee, datos, contenedor,self.__Heuristic)
        if newBee.fi > ColonyWorker[i].fi:
            ColonyWorker[i] = newBee
            self.fail[i][1] =0
        else:
            self.fail[i][1] += 1
 
    def ImproveB(self,i: int, ColonyWorker: List[Ind]):
        return MutateC2(ColonyWorker[i].genome)

    def WorkerBeePhase(self,ColonyWorker: List[Ind],datos:List[List[int]],contenedor:List[int])->None:
        for i in np.arange(self.pop_num):
            self.ImproveFlower(i,ColonyWorker,datos,contenedor)

    def OnlookerPhase(self,ColonyWorker: List[Ind],datos:List[List[int]],contenedor:List[int])->None:
        for _ in np.arange(self.pop_num):
            self.ImproveFlower(Tournament(ColonyWorker,0.85),ColonyWorker,datos,contenedor)

    def ScoutPhase(self,ColonyWorker: List[Ind],datos:List[List[int]],contenedor:List[int])->None:
        maxFail = self.fail.copy()
        maxFail.sort(key=lambda x: x[1], reverse=True)
        if maxFail[0][1] >= self.Limit:
            nwgn = np.asarray(np.random.choice(self.n,self.n,replace=False),dtype=np.int64)+1
            edabc = create_intidivual(NumbaList(nwgn))
            CalcFi(edabc,datos,contenedor,self.__Heuristic)
            ColonyWorker[maxFail[0][0]] = edabc
            if ColonyWorker[maxFail[0][0]].fi > self.BestInd.fi:
                self.BestInd = ColonyWorker[maxFail[0][0]] 
            self.fail[maxFail[0][0]][1] = 0

    def SelectHeuristic(self, hID:int)->None:
        self.__Heuristic= hID

EABC_type.define(EDABC.class_type.instance_type)
@njit 
def createEDABC(heuristic:int=0)->EDABC:
    return EDABC(heuristic)