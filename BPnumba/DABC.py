import numpy as np
from numba.typed import List as NumbaList
from BPnumba.GeneticOperators import RouletteWheel,Tournament, ind_type,SwapPointValue,create_intidivual,CalcFi,Ind,RRIS
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
        xj = round(xmin + np.random.uniform(0, 1)*(xmax-1))
        if xj <= xmin:
            xj = xmin
            xmin += 1
        elif xj >= xmax:
            xj = xmax
            xmax -= 1
        while visited[xj]:
            xj = round(xmin + np.random.uniform(0, 1)*(xmax-1))
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
specABC['fail'] = types.ListType(types.ListType(types.int64))
specABC['bestfi'] = types.ListType(types.float64)
specABC['__Heuristic'] = types.int64

@jitclass(specABC)
class DABC:
    def __init__(self, pop_num: int, n: int,heuristic:int=0):
        self.pop_num = pop_num
        self.n = n #Numero de cajas = max numero eWntero de ID de caja
        self.BestInd = Ind(NumbaList([1]))
        self.Limit = pop_num*n*100
        listaL = [ NumbaList([i,0]) for i in np.arange(pop_num)]
        self.fail:List[List[int]] = NumbaList(listaL)
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64))
        self.__Heuristic = heuristic

    def Train(self,numItr: int, ColonyWorker: List[Ind], datos:List[List[int]], contenedor:List[int]):
        rd :List[float]= []
        self.pop_num = len(ColonyWorker)
        self.n = len(datos)
        listaL = [ NumbaList([i,0]) for i in np.arange(self.pop_num)]
        self.fail:List[List[int]] = NumbaList(listaL)
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64))
        self.Limit = self.pop_num*self.n
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

            rd.append(self.BestInd.fi)
            if self.BestInd.fi == 1:
                break
        rd = np.array(rd,dtype=np.float64)
        self.bestfi = NumbaList(rd)
        return self.BestInd

    def ImproveFlower(self,i:int,ColonyWorker:List[Ind],datos:List[List[int]],contenedor:List[int]):
        nwGen =self.ImproveB(i, ColonyWorker)
        newBee = create_intidivual(nwGen)
        CalcFi(newBee, datos, contenedor,self.__Heuristic)
        if newBee.fi > ColonyWorker[i].fi:
            ColonyWorker[i] = newBee
            self.fail[i][1] =0
        else:
            self.fail[i][1] += 1
        if ColonyWorker[i].fi >self.BestInd.fi:
            self.BestInd = ColonyWorker[i]      

    def ImproveB(self,i: int, ColonyWorker: List[Ind]) -> List[int]:
        k = random.randint(0, self.n-1)
        j = random.randint(0, self.pop_num-1)
        while i == j:
            j = random.randint(0, self.pop_num-1)
        beeJ = ColonyWorker[j]
        nwcd = ColonyWorker[i].genome.copy()
        vik = abs(round(ColonyWorker[i].genome[k] + random.uniform(-1, 1)
                * (ColonyWorker[i].genome[k]-beeJ.genome[k])))
        if vik > self.n:
            vik = self.n
        elif vik < 1:
            vik = 1
        SwapPointValue(nwcd,k,vik)
        return nwcd

    def WorkerBeePhase(self,ColonyWorker:List[Ind],datos:List[List[int]],contenedor:List[int]):
        for i in np.arange(self.pop_num):
            self.ImproveFlower(i,ColonyWorker,datos,contenedor)
    
    def OnlookerPhase(self,ColonyWorker:List[Ind],datos:List[List[int]],contenedor:List[int]):
        for _ in np.arange(self.pop_num):
            j = RouletteWheel(ColonyWorker)
            self.ImproveFlower(j,ColonyWorker,datos,contenedor)
    def ScoutPhase(self,ColonyWorker:List[Ind],datos:List[List[int]],contenedor:List[int]):
        maxFail = self.fail.copy()
        maxFail.sort(key=lambda x: x[1], reverse=True)
        if maxFail[0][1] >= self.Limit:
            ColonyWorker[maxFail[0][0]] = CreateUniformRandomSoltion(self.n, datos, contenedor)
            self.fail[maxFail[0][0]][1] = 0
    def SelectHeuristic(self, hID:int)->None:
        self.__Heuristic= hID
ABC_type.define(DABC.class_type.instance_type)
@njit 
def createDABC(pop_num: int, n: int,heuristic:int=0):
    return DABC(pop_num,n,heuristic)
    

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
    def __init__(self, pop_num: int, n: int,heuristic:int=0):
        self.pop_num = pop_num
        self.n = n #Numero de cajas = max numero eWntero de ID de caja
        self.BestInd = Ind(NumbaList([1]))
        self.Limit = pop_num*n
        listaL = [ NumbaList([i,0]) for i in np.arange(pop_num)]
        self.fail:List[List[int]] = NumbaList(listaL)
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64))
        self.__Heuristic = heuristic

    def Train(self,numItr: int, ColonyWorker: List[Ind], datos:List[List[int]], contenedor:List[int])->Ind:
        rd :List[float]= []
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64))
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
            rd.append(self.BestInd.fi)
            if self.BestInd.fi == 1:
                break
        rd = np.array(rd,dtype=np.float64)
        self.bestfi = NumbaList(rd)
        return self.BestInd

    def ImproveFlower(self,i:int,ColonyWorker:List[Ind],datos:List[List[int]],contenedor:List[int])->None:
        nwGen =self.ImproveB(i, ColonyWorker)
        newBee = create_intidivual(nwGen)
        CalcFi(newBee, datos, contenedor,self.__Heuristic)
        if newBee.fi > ColonyWorker[i].fi:
            ColonyWorker[i] = newBee
            self.fail[i][1] =0
        else:
            self.fail[i][1] += 1
        if ColonyWorker[i].fi >self.BestInd.fi:
            self.BestInd = ColonyWorker[i]
        

    def ImproveB(self,i: int, ColonyWorker: List[Ind]) -> List[int]:
        n=self.n
        init = np.random.randint(1,n-2)
        end =  np.random.randint(init+1,n-1)
        r=np.random.random()
        if r<0.5:
            index=np.random.randint(0,init)
        else:
           index=np.random.randint(end+1,n)    
        return RRIS(ColonyWorker[i].genome,index,init,end)

    def WorkerBeePhase(self,ColonyWorker: List[Ind],datos:List[List[int]],contenedor:List[int])->None:
        for i in np.arange(self.pop_num):
            self.ImproveFlower(i,ColonyWorker,datos,contenedor)
    
    def OnlookerPhase(self,ColonyWorker: List[Ind],datos:List[List[int]],contenedor:List[int])->None:
        for _ in np.arange(self.pop_num):
            j = Tournament(ColonyWorker,0.9)
            self.ImproveFlower(j,ColonyWorker,datos,contenedor)

    def ScoutPhase(self,ColonyWorker: List[Ind],datos:List[List[int]],contenedor:List[int])->None:
        maxFail = self.fail.copy()
        maxFail.sort(key=lambda x: x[1], reverse=True)
        if maxFail[0][1] >= self.Limit:
            ColonyWorker[maxFail[0][0]] = CreateUniformRandomSoltion(self.n, datos, contenedor)
            self.fail[maxFail[0][0]][1] = 0
    def SelectHeuristic(self, hID:int)->None:
        self.__Heuristic= hID
EABC_type.define(EDABC.class_type.instance_type)
@njit 
def createEDABC(pop_num: int, n: int,heuristic:int=0)->EDABC:
    return EDABC(pop_num,n,heuristic)