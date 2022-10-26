import numpy as np
import random
from numba.typed import List as NumbaList
from numba import types, njit,deferred_type,jit,cuda,prange,vectorize
from numba.experimental import jitclass
from typing import List
from collections import OrderedDict
from BPnumba.GeneticOperators import CrossOX,MutateC1,MutateC2,MutateInversion
from BPnumba.Selection import Tournament,RouletteWheel
from BPnumba.Individual import Ind, ind_type,create_intidivual,createR_intidivual,CalcFi
from BPnumba.BoxN import ItemBin
import math


specAG = OrderedDict()
specAG['_prSelect'] = types.float64
specAG['_prMut'] = types.float64
specAG['_prMutR'] = types.float64

specAG['_prCross'] = types.float64
specAG['BestInd'] = ind_type
specAG['bestfi'] = types.ListType(types.float64)
specAG['__Rot'] = types.int64
specAG['__MutType'] = types.int64
@jitclass(specAG)
class NAG:
    def __init__(self,ps:float,pc:float,pmr:float,pm:float=0,adaptive=False,mutationType:int=0):
        self._prSelect=ps
        self._prCross =pc
        self._prMutR=pmr
        if pm != 0 and not adaptive:
            self._prMut=pm
        elif pm ==0 and adaptive:
            self._prMut=1.0
        else:
            raise Exception("Parameter of GA. pm is 0 iif adaptive is True.")
        self.BestInd = create_intidivual(NumbaList([1]))
        self.bestfi:List[float] = NumbaList(np.zeros(1,dtype=np.float64))
        self.__Rot = 0
        self.__MutType = mutationType

    def Train(self,maxItr:int,pob:List[Ind],datos:List[ItemBin],contenedor:List[int],rotType:int=0)->Ind:
        max_pop = len(pob)
        self.__Rot=rotType
        self.bestfi:List[float] = NumbaList(np.ones(maxItr,dtype=np.float64))
        pob.sort(key=lambda  ind : ind.fi, reverse=True)
        
        for _ in np.arange(maxItr):
            pob = self.NextGen(pob,rotType,datos,contenedor)
            self.bestfi[_]=pob[0].fi
            if pob[0].fi == 1 or (pob[0].fi-pob[max_pop-1].fi)/(pob[0].fi**2) <=0.001:
                if( _ != maxItr-1):
                    for _ in np.arange(_+1,maxItr):
                        self.bestfi[_]=pob[0].fi
                break
        self.BestInd=pob[0]
        return self.BestInd
    
    
    def NextGen(self,pob:List[Ind],rot:int,datos:List[ItemBin],contenedor:List[int])->List[Ind]:
        n = len(pob)
        k = np.random.randint(n/4,n/2)
        nwgn:list[Ind] = pob[:k]

        while len(nwgn)<n:
            id1 = self.Selection(pob)
            id2 = self.Selection(pob)
            while id2 == id1 :
                id2 = self.Selection(pob)
 

            if np.random.random() <= self._prCross:
                pm:float = 0.0
                if self._prMut == 1.0:
                    pm = 1.0-(pob[id1].fi+pob[id2].fi)/2
                else:
                    pm = self._prMut
                    
                child1 = MakeChild(f1=pob[id1],
                          f2=pob[id2],
                          MutType = self.__MutType,
                          pm=pm,
                          pmr=self._prMutR,
                          boxes=datos,
                          container=contenedor,
                          Rot=rot)
                
                child2 = MakeChild(f1=pob[id2],
                          f2=pob[id1],
                          MutType = self.__MutType,
                          pm=pm,
                          pmr=self._prMutR,
                          boxes=datos,
                          container=contenedor,
                          Rot=rot)
                nwgn.append(child1)
                if child1.fi != child2.fi:
                    nwgn.append(child2)
        nwgn.sort(key=lambda ind : ind.fi, reverse=True)
        return nwgn[:n]

    def Selection(self,poblation:List[Ind])->int:
        return Tournament(poblation,self._prSelect)

    def Elitism(self,pob:List[Ind],bestNum:int)->List[Ind]:
        return pob[:bestNum]
@njit
def createAG(ps:float,pc:float,pmr:float,pm:float=0,adaptive=False,mutType:int=0):
    return NAG(ps,pc,pmr,pm,adaptive,mutType)
ag_type = deferred_type()
ag_type.define(NAG.class_type.instance_type)


@njit
def GASearch(maxItr:int,ps:float,pc:float,pmr:float,mut:int,pob:List[Ind],datos:List[ItemBin],contenedor:List[int],rotType:int=0)->Ind:
    max_pop = len(pob)
    pob.sort(key=lambda  ind : ind.fi, reverse=True)
    for _ in np.arange(maxItr):
        NxGen(
            prSelect=ps,
            prCross=pc,
            prMutR=pmr,
            MutType=mut,
            pob=pob,
            rot=rotType,
            datos=datos,
            contenedor=contenedor)
        pob.sort(key=lambda  ind : ind.fi, reverse=True)
        #bestfi[_]=pob[0].fi
        if pob[0].fi == 1 or (pob[0].fi-pob[max_pop-1].fi)/(pob[0].fi**2) <=0.001:
            #if( _ != maxItr):
            #    for k in np.arange(_+1,maxItr):
            #        bestfi[k]=pob[0].fi
            break
    return pob[0]

@njit
def NxGen(prSelect,prCross,prMutR,MutType,pob:List[Ind],rot:int,datos:List[ItemBin],contenedor:List[int]):
        n = len(pob)
        k = np.random.randint(n/4,n/2)
        prevPob= pob.copy()
        for l in np.arange(k,n):
            id1 = Tournament(prevPob,prSelect)
            id2 = Tournament(prevPob,prSelect)
            while id2 == id1 :
                id2 = Tournament(prevPob)
            if np.random.random() <= prCross:
                pm = 1.0-(prevPob[id1].fi+prevPob[id2].fi)/2
            child1 =MakeChild(f1=prevPob[id1],
                          f2=prevPob[id2],
                          MutType = MutType,
                          pm=pm,
                          pmr=prMutR,
                          boxes=datos,
                          container=contenedor,
                          Rot=rot)
                
            child2 = MakeChild(f1=prevPob[id2],
                f2=prevPob[id1],
                MutType = MutType,
                pm=pm,
                pmr=prMutR,
                boxes=datos,
                container=contenedor,
                Rot=rot)
            if child1.fi >  child2.fi:
                pob[l]=child1
            else:
                pob[l]=child2
@njit
def MakeChild(f1:Ind,f2:Ind,MutType:float,pm:float,pmr:float,boxes:List[ItemBin],container:List[int],Rot:int)->Ind:
    cx = Crossover(f1,f2)
    cx =Mutation(MutType, NumbaList(cx[0]),NumbaList(cx[1]),pm)
    if Rot !=0:
        FlipMutation(boxes=boxes,gen=cx[0],rotgen=cx[1],pm=pmr,rotType=Rot)    
    ind = createR_intidivual(NumbaList(cx[0]),NumbaList(cx[1]))
    CalcFi(ind,boxes,NumbaList(container),Rot)
    return ind

@njit
def Mutation(MutType:int,gene:List[int],rgen:List[int], pm:float)->List[List[int]]:
    if random.random() > pm:
        return NumbaList([gene.copy(),rgen.copy()])
    if MutType==1:
        resp= NumbaList(MutateC1(NumbaList(gene),NumbaList(rgen)))
        return resp
    elif MutType==2:
        return NumbaList(MutateC2(NumbaList(gene),NumbaList(rgen)))
    else:
        return NumbaList(MutateInversion(NumbaList(gene),NumbaList(rgen)) )
@njit
def FlipMutation(boxes:List[ItemBin],gen:List[int],rotgen:List[int], pm:float,rotType:int):
    N = len(gen)  
    a = np.random.random(size=N)
    for i in np.arange(N):
        if a[i]<=pm:
            box=boxes[gen[i]-1]
            if rotgen[i] >= rotType-1:
                rotgen[i]=0
            else:
                rotgen[i]+=1
            box.rotate(rotgen[i])
            while not box.isValidRot():
                if rotgen[i] >= rotType-1:
                        rotgen[i]=0
                else:
                    rotgen[i]+=1
                box.rotate(rotgen[i])
@njit
def Crossover(ind1:Ind,ind2:Ind)->List[List[int]]:
    n = len(ind1.genome)
    i= random.randrange(3,int(n/2))
    j= random.randrange(i+1,n)
    resp = CrossOX(ind1.genome,ind2.genome,ind1.genome_r,ind2.genome_r,i,j)        
    return resp

@vectorize
def flip(x, y,pm,rot):
    if x > pm:
        return y
    else:
        if y == rot-1:
            return 0
        else: 
            return y+1

def makeChild(f1:Ind,f2:Ind,MutType:float,pm:float,pmr:float,boxes:List[ItemBin],container:List[int],Rot:int)->Ind:
    cx = Crossover(f1,f2)
    cx =Mutation(MutType, NumbaList(cx[0]),NumbaList(cx[1]),pm)
    if Rot !=0:
        cudaflip(gen=np.array(cx[1],dtype=np.int64),pm=pmr,rotType=Rot)    
    ind = createR_intidivual(NumbaList(cx[0]),NumbaList(cx[1]))
    CalcFi(ind,boxes,NumbaList(container),Rot)
    return ind

@jit
def cudaflip(gen, pm:float,rotType:int):
    N = len(gen)  
    a = np.random.random(size=N)
    dev_a = cuda.to_device(a)
    dev_b = cuda.to_device(gen)
    threads_per_block = 1024
    blocks_per_grid = (N + (threads_per_block - 1)) 
    dev_c = cuda.device_array_like(gen)
    flipCuda[blocks_per_grid, threads_per_block](dev_a, dev_b, dev_c,pm,rotType)
    gen = dev_c.copy_to_host()
    

@cuda.jit
def flipCuda(a, b, c,pm,rot):
    i = cuda.grid(1)
    if i < a.size:
        if a[i]>pm:
            c[i] = b[i]
        else:
            if c[i]==rot:
                c[i] = 0
            else:
                c[i] = b[i]+1     



@njit
def NotRepeated(h1:str,existedPob:List[str]):
    for j in np.arange(len(existedPob)):
        if h1 == existedPob[j]:
           return False
    return True



@njit
def factorial(n):
    return math.gamma(n+1)
@njit
def binomial(n,p,x):
    return (factorial(n)*np.power(p,x)*np.power(1-p,n-x))/(factorial(x)*factorial(n-x))
