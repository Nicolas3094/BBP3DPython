import numpy as np
import random
from numba.typed import List as NumbaList
from numba import types, njit,deferred_type
from numba.experimental import jitclass
from typing import List
from collections import OrderedDict
from BPnumba.GeneticOperators import CrossOX,MutateC1,MutateC2,MutateInversion
from BPnumba.Selection import Tournament
from BPnumba.Individual import Ind, ind_type,create_intidivual,CalcFi,CodeSolution



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
            self._prMut=1.0
        else:
            raise Exception("Parameter of GA. pm is 0 iif adaptive is True.")
        self.BestInd = Ind(NumbaList([1]))
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64))
        self.__Heuristic = heuristic
        self.__MutType = mutationType

    def Train(self,maxItr:int,pob:List[Ind],datos:List[List[int]],contenedor:List[int])->Ind:
        max_pop = len(pob)
        self.bestfi:List[float] = NumbaList(np.ones(maxItr,dtype=np.float64))
        pob.sort(key=lambda  ind : ind.fi, reverse=True)
        for _ in np.arange(maxItr):
            pob = self.NextGen(pob,datos,contenedor)
            self.bestfi[_]=pob[0].fi
            if pob[0].fi == 1:
                break
            if (pob[0].fi-pob[max_pop-1].fi)/(pob[0].fi**2) <=0.001:
                for _ in np.arange(_+1,maxItr):
                    self.bestfi[_]=pob[0].fi
                break
        self.BestInd=pob[0]
        return self.BestInd

    def NextGen(self,pob:List[Ind],datos,contenedor)->List[Ind]:
        n = len(pob)
        k = np.random.randint(n/4,n/2)
        visitedParent = list(np.arange(k))
        nwgn:list[Ind] = pob[:k]

        while len(nwgn)<n:
            id1 = self.Selection(pob)
            id2 = self.Selection(pob)
            while id2 == id1 :
                id2 = self.Selection(pob)
            if id1 not in visitedParent:
                nwgn.append(pob[id1])
                visitedParent.append(id1)
            if id2 not in visitedParent:
                nwgn.append(pob[id2])
                visitedParent.append(id2)
            if np.random.random() <= self._prCross:
                pm:float = 0.0
                if self._prMut == 1.0:
                    pm = 1.0-(pob[id1].fi+pob[id2].fi)/2
                else:
                    pm = self._prMut
                h1 = self.Crossover(pob[id1],pob[id2])

                h1=self.Mutation(NumbaList(h1),pm)
                
                if h1 != pob[id1].genome and h1 != pob[id2].genome:
                    ind1 = create_intidivual(NumbaList(h1))
                    CalcFi(ind1,NumbaList(datos),NumbaList(contenedor),self.__Heuristic)
                    nwgn.append(ind1)

                h2 = self.Crossover(pob[id2],pob[id1])
                h2=self.Mutation(NumbaList(h2),pm)
                if h2 != pob[id1].genome and h2 != pob[id2].genome:
                    ind2 = create_intidivual(NumbaList(h2))
                    CalcFi(ind2,NumbaList(datos),NumbaList(contenedor),self.__Heuristic)
                    nwgn.append(ind2)
        nwgn.sort(key=lambda ind : ind.fi, reverse=True)
        return nwgn[:n]

    def Selection(self,poblation:List[Ind])->int:
        return Tournament(poblation,self._prSelect)
    def Crossover(self,ind1:Ind,ind2:Ind)->List[int]:
        n = len(ind1.genome)
        i= random.randrange(3,int(n/2))
        j= random.randrange(i+1,n)
        resp = CrossOX(ind1.genome,ind2.genome,i,j)
        return resp
    def MutationRange(self,gene:List[int], pm) -> List[int]:
        n:int=len(gene)
        if self.__MutType==1:
            return MutateC1(NumbaList(gene),round(pm*n))
        elif self.__MutType==2:
            return MutateC2(NumbaList(gene),round(pm*n))
        else:
            return MutateInversion(NumbaList(gene),round(pm*n))
    def Mutation(self,gene:List[int], pm:float)->List[int]: 
            if random.random() > pm:
                return gene.copy()
            if self.__MutType==1:
                return MutateC1(NumbaList(gene))
            elif self.__MutType==2:
                return MutateC2(NumbaList(gene))
            else:
                return MutateInversion(NumbaList(gene))          
          
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

