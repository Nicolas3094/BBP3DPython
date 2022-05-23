import numpy as np
from numba.typed import List as NumbaList
from BPnumba.NumAG import CalcFi, create_intidivual, Ind
from BPnumba.GeneticOperators import RouletteWheel, ind_type,SwapPointValue
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
        xj = round(xmin + random.uniform(0, 1)*(xmax-1))
        if xj <= xmin:
            xj = xmin
            xmin += 1
        elif xj >= xmax:
            xj = xmax
            xmax -= 1
        while visited[xj]:
            xj = round(xmin + random.uniform(0, 1)*(xmax-1))
            if xj <= xmin:
                xj = xmin
                xmin += 1
            elif xj >= xmax:
                xj = xmax
                xmax -= 1
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
specABC['fail'] = types.ListType(types.int64)
specABC['bestfi'] = types.ListType(types.float64)
@jitclass(specABC)
class DABC:
    def __init__(self, pop_num: int, n: int):
        self.pop_num = pop_num
        self.n = n #Numero de cajas = max numero eWntero de ID de caja
        self.BestInd = Ind(NumbaList([1]))
        self.Limit = pop_num*n*100
        fail = np.zeros(pop_num, dtype=np.int64)
        self.fail = NumbaList(fail)
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64))

    def Train(self,numItr: int, ColonyWorker: List[Ind], datos:List[List[int]], contenedor:List[int]):
        rd :List[float]= []
        self.pop_num = len(ColonyWorker)
        self.n = len(datos)
        self.fail = NumbaList(np.zeros(len(ColonyWorker), dtype=np.int64))
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64))
        self.Limit = self.pop_num*(len(datos))*100

        for _ in np.arange(numItr):
            # Busqueda local de abejas para todas las trabajadoras y calcila Pi
            self.WorkerBeePhase(ColonyWorker,datos,contenedor)
            # Busqueda Onlooker bee
            self.OnlookerPhase(ColonyWorker,datos,contenedor)
            #Busqueda de mejoras de acuerdo al Limite
            self.ScoutPhase(ColonyWorker,datos,contenedor)
            rd.append(self.BestInd.fi)
            if self.BestInd.fi == 1:
                break
        rd = np.array(rd,dtype=np.float64)
        self.bestfi = NumbaList(rd)
        return self.BestInd

    def ImproveFlower(self,i:int,ColonyWorker:List[Ind],datos,contenedor):
        pos = ColonyWorker[i].genome.copy()
        fi = ColonyWorker[i].fi
        sol = ColonyWorker[i].codeSolution
        load = ColonyWorker[i].load
        self.ImproveB(i, ColonyWorker)
        CalcFi(ColonyWorker[i], datos, contenedor)
        if fi > ColonyWorker[i].fi:
            self.fail[i] += 1
            ColonyWorker[i].genome = pos
            ColonyWorker[i].codeSolution=sol
            ColonyWorker[i].load=load
            ColonyWorker[i].fi=fi
        else:
            self.fail[i] = 0
        if self.BestInd.fi < ColonyWorker[i].fi:
            self.BestInd = ColonyWorker[i]

    def ImproveB(self,i: int, ColonyWorker: List[Ind]) -> None:
        k = random.randint(0, self.n-1)
        j = random.randint(0, self.pop_num-1)
        while i == j:
            j = random.randint(0, self.pop_num-1)
        beeK = ColonyWorker[j]
        vik = abs(round(ColonyWorker[i].genome[k] + random.uniform(-1, 1)
                * (ColonyWorker[i].genome[k]-beeK.genome[k])))
        if vik > self.n:
            vik = self.n
        elif vik < 1:
            vik = 1
        ColonyWorker[i].genome[k] = vik
        SwapPointValue(ColonyWorker[i].genome,k,vik)

    def WorkerBeePhase(self,ColonyWorker,datos,contenedor):
        for i in np.arange(self.pop_num):
            self.ImproveFlower(i,ColonyWorker,datos,contenedor)
    
    def OnlookerPhase(self,ColonyWorker,datos,contenedor):
        for _ in np.arange(self.pop_num):
            j = RouletteWheel(ColonyWorker)
            self.ImproveFlower(j,ColonyWorker,datos,contenedor)
    def ScoutPhase(self,ColonyWorker,datos,contenedor):
        maxFail=0
        maxIdex = 0
        for k in np.arange(len(self.fail)):#puede mejorar busqueda
            if self.fail[k]>maxFail:
                maxFail=self.fail[k]
                maxIdex=k
        if maxFail >= self.Limit:
            ColonyWorker[maxIdex] = CreateUniformRandomSoltion(self.n, datos, contenedor)
            maxFail = 0
            self.fail[maxIdex] = 0
ABC_type.define(DABC.class_type.instance_type)
@njit 
def createDABC(pop_num: int, n: int):
    return DABC(pop_num,n)