import numpy as np
import random
from numba.typed import List as NumbaList
from numba import types, prange,njit,deferred_type
from typing import List
from collections import OrderedDict
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from BPnumba.GeneticOperators import Ind, Tournament,CrossOX,InverseMutation,create_intidivual,CalcFi,Hamming


specAG = OrderedDict()
specAG['_prSelect'] = types.float64
specAG['_prMut'] = types.float64
specAG['_prCross'] = types.float64
@jitclass(specAG)
class NAG:
    def __init__(self,ps:float,pm:float,pc:float):
        self._prSelect=ps
        self._prMut =pm
        self._prCross=pc
    @staticmethod
    def Selection(poblation:List[Ind],pr_selecton:float)->int:
        return Tournament(poblation,pr_selecton)
    @staticmethod
    def Crossover(ind1:Ind,ind2:Ind)->List[int]:
        r = random.random()
        n = len(ind1.genome)
        i= random.randrange(3,int(n/2))
        j= random.randrange(i+1,n)
        resp = CrossOX(ind1.genome,ind2.genome,i,j)
        return resp
    @staticmethod
    def Mutation(gene:List[int],prmUt):
        r = random.random()
        if r <=  prmUt:
            n = len(gene)
            i= random.randrange(1,int(n/2))
            j= random.randrange(i+1,n)
            InverseMutation(NumbaList(gene),i,j)
    @staticmethod
    def Elitism(pob:List[Ind],bestNum:int)->List[Ind]:
        return pob[:bestNum]       
        
@njit(nogil=True)
def create_AG(ps:float,pm:float,pc:float):
    return NAG(ps,pm,pc)
ag_type = deferred_type()
ag_type.define(NAG.class_type.instance_type)

@njit(parallel=True)
def CreateNexGen(GA:NAG,pob:List[Ind],datos,contenedor):
    n = len(pob)
    k = random.randint(int(n/4),n) #numero de individuos intentar por crear por pares
    existedPob = [ind.genome for ind in pob]
    if k % 2 != 0:
         k -=1 
    for _ in np.arange(k):
        id1 = GA.Selection(pob,GA._prSelect)
        id2 = GA.Selection(pob,GA._prSelect)
        while id2 == id1:
            id2 = GA.Selection(pob,GA._prSelect)
        rn = random.random()
        if rn <= GA._prCross:
            rp = random.random()
            if GA._prMut<=rp:
                h1 = GA.Crossover(pob[id1],pob[id2])
                h2 = GA.Crossover(pob[id2],pob[id1])
                foo =True
                foo2=True
                for j in prange(len(existedPob)):
                    if Hamming(NumbaList(h1),existedPob[j]) == 0 and foo:
                        foo = False
                        break
                for j in prange(len(existedPob)):
                    if Hamming(NumbaList(h2),existedPob[j]) == 0 and foo2:
                        foo2 = False
                        break                        
                if foo:
                    ind1 = create_intidivual(NumbaList(h1))
                    CalcFi(ind1,NumbaList(datos),NumbaList(contenedor))
                    pob.append(ind1)
              
                if foo2:
                    ind2 = create_intidivual(NumbaList(h2))
                    CalcFi(ind2,NumbaList(datos),NumbaList(contenedor))
                    pob.append(ind2)
        pob.sort(key=lambda ind : ind.fi, reverse=True)

@njit
def TrainAG(maxItr:int,pob:List[Ind],pt:float,pc:float,pr:float,datos,contenedor):
    ag=create_AG(pt,pr,pc)
    max_pop = len(pob)
    pob.sort(key=lambda  ind : ind.fi, reverse=True)
    for _ in np.arange(maxItr):
        CreateNexGen(ag,pob,datos,contenedor)
        pob = ag.Elitism(pob,max_pop)
        if pob[0].fi == 1 or (pob[0].fi-pob[-1].fi)/(pob[0].fi**2)<0.001:
            break
    return pob[0]


