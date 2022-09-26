import numpy as np
from numba.typed import List as NumbaList
from BPnumba.GeneticOperators import Tournament, ind_type,SwapPointValue,create_intidivual,CalcFi,Ind
from numba import njit, deferred_type, types
from typing import List
import random
from numba.experimental import jitclass
from collections import OrderedDict
@njit
def CreateUniformRandomSoltion(n: int, boxes: List[List[int]], container: List[int]) -> Ind:
    xmin = 1
    xmax = n
    visited = [False for i in np.arange(n+1)]
    newcode = np.zeros(n, dtype=np.int64)
    for i in np.arange(n):
        xj = abs(round(xmin + np.random.uniform(0, 1)*(xmax-1)))
        if xj > xmax:
            xj -= xmin
        if xj == xmax:
            xmax-=1
        if xj < xmin:
            xj=xmin
        if xj==xmin:
            xmin +=1
        while visited[xj]:
            xj = abs(round(xmin + np.random.uniform(0, 1)*(xmax-1)))
            if xj > xmax:
                xj -= xmin
            if xj == xmax:
                xmax-=1
            if xj < xmin:
                xj=xmin
            if xj==xmin:
                xmin +=1
        newcode[i] = xj
        visited[xj] = True
    nwbee = create_intidivual(NumbaList(newcode))
    CalcFi(nwbee, boxes, container)
    return nwbee 

ABC_type = deferred_type()
specABC = OrderedDict()
specABC['pop_num'] = types.int64
specABC['n'] = types.int64
specABC['BestInd'] = ind_type
specABC['Limit'] = types.int64
specABC['fail'] = types.ListType(types.ListType(types.int64))
specABC['bestfi'] = types.ListType(types.float64)
specABC['__Heuristic'] = types.int64
@jitclass(specABC)
class DABC:
    def __init__(self,heuristic:int=0):
        self.pop_num = 1
        self.n = 1 #Numero de cajas = max numero eWntero de ID de caja
        self.BestInd = Ind(NumbaList([1]))
        self.Limit = 1
        listaL = [ NumbaList([i,0]) for i in np.arange(1)]
        self.fail:List[List[int]] = NumbaList(listaL)
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64))
        self.__Heuristic = heuristic

    def Train(self,numItr: int, ColonyWorker: List[Ind], datos:List[List[int]], contenedor:List[int]):
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
            # Busqueda local de abejas para todas las trabajadoras 
            self.WorkerBeePhase(ColonyWorker=ColonyWorker,datos=datos,contenedor=contenedor)
            # Busqueda Onlooker bee
            self.OnlookerPhase(ColonyWorker=ColonyWorker,datos=datos,contenedor=contenedor)
            #Busqueda de mejoras de acuerdo al Limite
            self.ScoutPhase(ColonyWorker=ColonyWorker,datos=datos,contenedor=contenedor)
            self.bestfi[_]=self.BestInd.fi
            if self.BestInd.fi == 1:
                break    

        return self.BestInd
    
    def ImproveFlower(self,i:int,ColonyWorker:List[Ind],datos:List[List[int]],contenedor:List[int]):
        nwGen = NumbaList(self.ImproveB(i, ColonyWorker))
        newBee = create_intidivual(NumbaList(nwGen))
        CalcFi(newBee, datos, contenedor,self.__Heuristic)
        if newBee.fi > ColonyWorker[i].fi:
            ColonyWorker[i] = newBee
            self.fail[i][1] =0
        else:
            self.fail[i][1] += 1
    
    def ImproveB(self,i: int, ColonyWorker: List[Ind]) -> List[int]:
        k = np.random.randint(0, self.n-1)
        j = np.random.randint(0, self.pop_num-1)
        while i == j:
            j = random.randint(0, self.pop_num-1)
        beeJ = ColonyWorker[j]
        nwcd = ColonyWorker[i].genome.copy()
        vik = abs(round(ColonyWorker[i].genome[k] + np.random.uniform(-1, 1)*(ColonyWorker[i].genome[k]-beeJ.genome[k])))
        if vik > self.n:
            vik -= self.n
        if vik < 1:
            vik = 1
        SwapPointValue(nwcd,k,vik)
        return nwcd

    def WorkerBeePhase(self,ColonyWorker:List[Ind],datos:List[List[int]],contenedor:List[int]):
        for i in np.arange(self.pop_num):
            self.ImproveFlower(i=i,ColonyWorker=ColonyWorker,datos=datos,contenedor=contenedor)

    def OnlookerPhase(self,ColonyWorker:List[Ind],datos:List[List[int]],contenedor:List[int]):
        for _ in np.arange(self.pop_num):
            j = Tournament(ColonyWorker,pt=0.85)
            self.ImproveFlower(j,ColonyWorker,datos,contenedor)
    def ScoutPhase(self,ColonyWorker:List[Ind],datos:List[List[int]],contenedor:List[int]):
        maxFail = self.fail.copy()
        maxFail.sort(key=lambda x: x[1], reverse=True)
        if maxFail[0][1] >= self.Limit:
            self.fail[maxFail[0][0]][1] = 0
            ColonyWorker[maxFail[0][0]] = CreateUniformRandomSoltion(self.n, datos, contenedor)
            if ColonyWorker[maxFail[0][0]].fi > self.BestInd.fi:
                self.BestInd= ColonyWorker[maxFail[0][0]]

    def SelectHeuristic(self, hID:int)->None:
        self.__Heuristic= hID

ABC_type.define(DABC.class_type.instance_type)
@njit 
def createDABC(heuristic:int=0):
    return DABC(heuristic)