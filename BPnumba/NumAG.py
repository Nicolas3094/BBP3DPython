import numpy as np
import random
from numba.typed import List as NumbaList
from numba import types, prange,njit,deferred_type
from typing import List
from collections import OrderedDict
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from BPnumba.GeneticOperators import Ind, ind_type,Tournament,CrossOX,InverseMutation,create_intidivual,CalcFi,Hamming
specAG = OrderedDict()
specAG['_prSelect'] = types.float64
specAG['_prMut'] = types.float64
specAG['_prCross'] = types.float64
specAG['BestInd'] = ind_type
specAG['bestfi'] = types.ListType(types.float64)

@jitclass(specAG)
class NAG:
    def __init__(self,ps:float,pm:float,pc:float):
        self._prSelect=ps
        self._prMut =pm
        self._prCross=pc
        self.BestInd = Ind(NumbaList([1]))
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64))

    def Train(self,maxItr:int,pob:List[Ind],datos,contenedor):
        max_pop = len(pob)
        rd :List[float]= []
        pob.sort(key=lambda  ind : ind.fi, reverse=True)
        for _ in np.arange(maxItr):
            self.NextGen(pob,datos,contenedor)
            pob = self.Elitism(pob,max_pop)
            rd.append(pob[0].fi)
            if pob[0].fi == 1 or (pob[0].fi-pob[-1].fi)/(pob[0].fi**2)<0.001:
                break
        self.BestInd=pob[0]
        rd = np.array(rd,dtype=np.float64)
        self.bestfi = NumbaList(rd)

    def NextGen(self,pob:List[Ind],datos,contenedor):
        n = len(pob)
        k = random.randint(int(n/4),n) #numero de individuos intentar por crear por pares
        existedPob = [ind.genome for ind in pob]
        if k % 2 != 0:
            k -=1 
        for _ in np.arange(k):
            id1 = self.Selection(pob)
            id2 = self.Selection(pob)
            while id2 == id1:
                id2 = self.Selection(pob)
            rn = random.random()
            if rn <= self._prCross:
                rp = random.random()
                if self._prMut<=rp:
                    h1 = self.Crossover(pob[id1],pob[id2])
                    h2 = self.Crossover(pob[id2],pob[id1])                  
                    if NotRepeated(h1,existedPob):
                        self.Mutation(NumbaList(h1))
                        ind1 = create_intidivual(NumbaList(h1))
                        CalcFi(ind1,NumbaList(datos),NumbaList(contenedor))
                        pob.append(ind1)
                    if NotRepeated(h2,existedPob):
                        self.Mutation(NumbaList(h2))
                        ind2 = create_intidivual(NumbaList(h2))
                        CalcFi(ind2,NumbaList(datos),NumbaList(contenedor))
                        pob.append(ind2)
            else:
                if pob[id1].fi > pob[id2].fi:
                    pob[id2].fi=0
                else:
                    pob[id1].fi=0
        pob.sort(key=lambda ind : ind.fi, reverse=True)
    def Selection(self,poblation:List[Ind])->int:
        return Tournament(poblation,self._prSelect)
    def Crossover(self,ind1:Ind,ind2:Ind)->List[int]:
        n = len(ind1.genome)
        i= random.randrange(3,int(n/2))
        j= random.randrange(i+1,n)
        resp = CrossOX(ind1.genome,ind2.genome,i,j)
        return resp
    def Mutation(self,gene:List[int]):
        r = random.random()
        if r <=  self._prMut:
            n = len(gene)
            i= random.randrange(1,int(n/2))
            j= random.randrange(i+1,n)
            InverseMutation(NumbaList(gene),i,j)
    def Elitism(self,pob:List[Ind],bestNum:int)->List[Ind]:
        return pob[:bestNum]  

@njit(parallel=True)
def NotRepeated(h1,existedPob):
    foo=True
    for j in prange(len(existedPob)):
        if Hamming(NumbaList(h1),existedPob[j]) == 0 and foo:
           return False
    return True

@njit
def create_AG(ps:float,pm:float,pc:float):
    return NAG(ps,pm,pc)
ag_type = deferred_type()
ag_type.define(NAG.class_type.instance_type)

