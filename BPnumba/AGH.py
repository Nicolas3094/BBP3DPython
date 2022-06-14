import numpy as np
import random
from numba.typed import List as NumbaList
from numba import types, njit,deferred_type
from numba.experimental import jitclass
from typing import List
from collections import OrderedDict
from BPnumba.GeneticOperators import Combine2,Ind, ind_type,Tournament,CrossOX,InverseMutation,create_intidivual,CalcFi,CodeSolution,Combine1

@njit
def MutateC2(genome: List[int], randomStep: int=0)->List[int]:
    n = int(len(genome))
    step=randomStep
    if randomStep==0:
        step = np.random.randint(1,n-2)
    init = np.random.randint(1,int(n/2))
    if init + step > n-2:
        end = n-2
    else:
        end =  np.random.randint(init+1,n-1)
    if np.random.random()<0.5:
        index=np.random.randint(0,init)
    else:
        index=np.random.randint(end+1,n)
    return Combine2(genome,index,init,end)

@njit
def MutateC1(gene:List[int],randomStep:int=0)->List[int]:
    n=len(gene)
    step=randomStep
    if randomStep==0:
        step = np.random.randint(1,n-2)
    i=np.random.randint(int(n/2)-int(step/2)+1)
    j = i + int(step/2)-1
    i2 = np.random.randint(int(n/2),n-int(step/2))
    j2 = i2 + int(step/2)-1
    return Combine1(NumbaList(gene),i,j,i2,j2)

@njit
def MutateInversion(gene:List[int])->List[int]:
    n=len(gene)
    i= random.randrange(1,int(n/2))
    j= random.randrange(i+1,n)
    return InverseMutation(NumbaList(gene),i,j)


specAG = OrderedDict()
specAG['_prSelect'] = types.float64
specAG['_prMut'] = types.float64
specAG['_prCross'] = types.float64
specAG['BestInd'] = ind_type
specAG['bestfi'] = types.ListType(types.float64)
specAG['__Heuristic'] = types.int64
specAG['__MutType'] = types.int64
@jitclass(specAG)
class NAG:
    def __init__(self,ps:float,pc:float,pm:float=0,heuristic:int =0,adaptive=False,mutationType:int=0):
        self._prSelect=ps
        self._prCross =pc
        if pm != 0 and not adaptive:
            self._prMut=pm
        elif pm ==0 and adaptive:
            self._prMut=1
        else:
            raise Exception("Parameter of GA. pm is 0 iif adaptive is True.")
        self.BestInd = Ind(NumbaList([1]))
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64))
        self.__Heuristic = heuristic
        self.__MutType = mutationType

    def Train(self,maxItr:int,pob:List[Ind],datos:List[List[int]],contenedor:List[int])->Ind:
        max_pop = len(pob)
        rd :List[float]= []
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64))
        self.BestInd = Ind(NumbaList([1]))

        pob.sort(key=lambda  ind : ind.fi, reverse=True)
        for _ in np.arange(maxItr):
            self.NextGen(pob,datos,contenedor)
            pob = self.Elitism(pob,max_pop)
            rd.append(pob[0].fi)
            if pob[0].fi == 1 or (pob[0].fi-pob[max_pop-1].fi)/(pob[0].fi**2) <=0.001:
                break
        self.BestInd=pob[0]
        rd = np.array(rd,dtype=np.float64)
        self.bestfi = NumbaList(rd)
        return self.BestInd

    def NextGen(self,pob:List[Ind],datos,contenedor):
        n = len(pob)
        k = np.random.randint(int(n/2),n) #numero de individuos intentar por crear por pares
        existedPob = [ind.codeSolution for ind in pob]
        if k % 2 != 0:
            k -=1
        for i in np.arange(k):
            if pob[i].codeSolution == pob[i+1].codeSolution:
               pob[i+1].fi=0
            id1 = self.Selection(pob)
            id2 = self.Selection(pob)
            while id2 == id1:
                id2 = self.Selection(pob)
            if pob[id1].codeSolution == pob[id2].codeSolution:
                if id1 < id2:
                    pob[id2].fi = 0
                else:
                    pob[id1].fi = 0
            if np.random.random() <= self._prCross:
                h1 = self.Crossover(pob[id1],pob[id2])
                h2 = self.Crossover(pob[id2],pob[id1])
                pm = 0
                if self._prMut == 1:
                    pm = 1-(pob[id1].fi+pob[id2].fi)/2
                else:
                    pm = self._prMut
                self.Mutation(NumbaList(h1),pm)
                ind1 = create_intidivual(NumbaList(h1))
                CalcFi(ind1,NumbaList(datos),NumbaList(contenedor),self.__Heuristic)
                self.Mutation(NumbaList(h2),pm)
                ind2 = create_intidivual(NumbaList(h2))
                CalcFi(ind2,NumbaList(datos),NumbaList(contenedor),self.__Heuristic)
                
                if ind1.codeSolution not in existedPob:
                    pob.append(ind1)
                if ind2.codeSolution not in existedPob:
                    pob.append(ind2)

        pob.sort(key=lambda ind : ind.fi, reverse=True)
    def Selection(self,poblation:List[Ind])->int:
        return Tournament(poblation,self._prSelect)
    def Crossover(self,ind1:Ind,ind2:Ind)->List[int]:
        n = len(ind1.genome)
        i= random.randrange(3,int(n/2))
        j= random.randrange(i+1,n)
        resp = CrossOX(ind1.genome,ind2.genome,i,j)
        return resp

    def Mutation(self,gene:List[int], pm:float):
        r = np.random.random()
        if r <= pm:
            if self.__MutType==1:
                gene=MutateC1(NumbaList(gene))
            elif self.__MutType==2:
                gene=MutateC2(NumbaList(gene))
            else:
                gene= MutateInversion(NumbaList(gene))          
          
    def Elitism(self,pob:List[Ind],bestNum:int)->List[Ind]:
        return pob[:bestNum]
          
    def SelectHeuristic(self, hID:int):
        self.__Heuristic= hID
@njit
def NotRepeated(h1:str,existedPob:List[str]):
    for j in np.arange(len(existedPob)):
        if h1 == existedPob[j]:
           return False
    return True

@njit
def createAG(ps:float,pc:float,pm:float=0,heuristic:int=0,adaptive=False,mutType:int=0):
    return NAG(ps,pc,pm,heuristic,adaptive,mutType)
ag_type = deferred_type()
ag_type.define(NAG.class_type.instance_type)

