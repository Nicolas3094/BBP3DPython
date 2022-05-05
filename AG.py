from audioop import reverse
from Poblacion import Poblation,Individuo
from Contenedor import Bin
import numpy as np
import random
from  OperatorG import Cruza,Cruza,Mutation
from heapq import heapify, heappush, heappop
from queue import PriorityQueue

class AG:
    def __init__(self, pc:float, pt:float, pm:float):
        self._pc = pc #prob cruza
        self._pt = pt #prob de torneo
        self._pm = pm #prob de mutacion
        self._DataSet=dict()
        self.pob:Poblation = None
        self.maxVolume= 1
        
    def __initValues__(self):
        self._gen = 0 
        self.fiBest = list()
        self.fiWorst = list()
        self.Convergencia = list()

    def _RegistrarDatos__(self):
        self.fiBest.append(self.pob[0].fi)
        self.fiWorst.append(self.pob[-1].fi)
    
    def PackingMethod(self, methodPakcing):
        self._packing=methodPakcing

    def Best(self):
        return self.pob[0]
    def Convergence(self)->float:
        return (self.Best().fi-self.pob[-1].fi)/(self.Best().fi**2)
    def RandomInitPob(self,N:int , BoxSeq:list, Heuristic:bool=False):
        Pob = Poblation(N)
        originalInd = [i for i in range(1,len(BoxSeq)+1)]
        self._DataSet = dict(zip(originalInd,BoxSeq)) #Data set Global en forma de diccionario, de [1,..n] -> [ p1, p2, ..., pn  ], donde pi = [li wi hi]
        if Heuristic:
            N -= 4
            self._maxVolume = 0
            for box in BoxSeq:
                self._maxVolume += np.prod(box)
            vol = list({k: v for k,v in sorted(self._DataSet.items(),key = lambda v: v[1][0]*v[1][1]*v[1][2])}.keys())
            Pob.CreateInd(code=vol ) #ordena por volumen
            for _ in range(3): #ordena por longitud, ancho y alto
                di = list({k: v for k,v in sorted(self._DataSet.items(),key = lambda v: v[1][_])}.keys())
                Pob.CreateInd(code=di)
        for _ in range(N):
            random.shuffle(originalInd)
            while originalInd in Pob:
                random.shuffle(originalInd)
            copyInd = originalInd.copy()
            Pob.CreateInd(code = copyInd) 
        self.pob = Pob
        self.__initValues__()

    def train(self,maxGen,contenedor,Data):
        self._maxVolume/= np.prod(contenedor)
        if self._maxVolume>1:
            self._maxVolume=1
        self.EvaluateFitness(self.pob,contenedor,Data)
        for _ in np.arange(maxGen):
            self._RegistrarDatos__()
            self.CrearGeneracion(contenedor,Data)
            if self.Condition():
                break
    def train2(self,maxGen,contenedor,Data):
        self.EvaluateFitness(self.pob,contenedor,Data)
        for _ in np.arange(maxGen):
            self._RegistrarDatos__()
            self.CrearGeneracion2(contenedor,Data)
            self.pob.poblation = self.pob.poblation[:self.pob.n]
            if self.Condition() or self.Convergence()<0.01:
                break

    def CrearGeneracion2(self,dimBin,DataSet):
        n = random.randint(int(self.pob.n/4),int(self.pob.n/2)) # maximo un cuarto de la mejor poblacion actual pasa a la siguiente , min 2 por elitismo, exento a cruza, los demas pasan a seleccion
        if n % 2 != 0:
            n -=1 
        newPob:list[Individuo] = []
        while len(newPob)<=n:
            indx1 = self.Seleccion(self.pob.poblation,self._pt)
            indx2 = self.Seleccion(self.pob.poblation,self._pt)
            while indx1 == indx2:
                indx2 = self.Seleccion(self.pob.poblation,self._pt)
            rn = random.random()
            if rn <= self._pc:
                p1 = self.pob.poblation[indx1]
                p2 = self.pob.poblation[indx2]
                h1,h2 = Cruza(p1.genome,p2.genome)
                Mutation(h1,self._pm)
                Mutation(h2,self._pm)
                if h1 in self.pob:
                    continue
                if h2 in self.pob:
                    continue
                ind1 = Individuo(code=h1)
                ind2 = Individuo(code=h2)
                self.EvalIndFit(ind1,dimBin,DataSet)
                self.EvalIndFit(ind2,dimBin,DataSet)
                newPob.append(ind1)
                newPob.append(ind2)
        self.pob.poblation = self.pob.poblation+newPob
        self.pob.poblation.sort(key=lambda x: x.fi,reverse=True)


    def EvalIndFit(self,ind:Individuo,dimBin,DataSet):
        bin:Bin = Bin(dimensiones=dimBin, n=len(DataSet))
        self._packing(bin=bin,itemsToPack=ind.genome, ITEMSDATA=DataSet)
        ind.fi = bin.getLoadVol()/np.prod(dimBin)

    def EvaluateFitness(self,pobl:Poblation,dimBin,DataSet):
        for individuo in pobl:
            if individuo.fi is None:
                bin:Bin = Bin(dimensiones=dimBin, n=len(DataSet))
                self._packing(bin=bin,itemsToPack=individuo.genome, ITEMSDATA=DataSet)
                individuo.fi = bin.getLoadVol()/np.prod(dimBin)
        pobl.poblation.sort(key= lambda x : x.fi,reverse=True)
    def CrearGeneracion(self,dimbin,dataset):
        n = random.randint(2,int(self.pob.n/4)) # maximo un cuarto de la mejor poblacion actual pasa a la siguiente , min 2 por elitismo, exento a cruza, los demas pasan a seleccion
        if n % 2 != 0:
            n -=1 
        worstPob:list[Individuo] = self.pob[n:]
        del self.pob.poblation[n:]
        while len(worstPob)!=0:
            indx1 = self.Seleccion(worstPob,self._pt)
            indx2 = self.Seleccion(worstPob,self._pt)
            while indx1 == indx2:
                indx2 = self.Seleccion(worstPob,self._pt)
            p1:Individuo = worstPob[indx1]
            p2:Individuo = worstPob[indx2]
            rn = random.random()
            if rn <= self._pc:
                h1,h2 = Cruza(p1.genome,p2.genome)
                probAdap = self._pm
                Mutation(h1,probAdap)
                Mutation(h2,probAdap)
                ind1 = Individuo(code=h1)
                ind2 = Individuo(code=h2)
                self.EvalIndFit(ind1,dimbin,dataset)
                self.EvalIndFit(ind2,dimbin,dataset)
                self.pob.Add(ind1)
                self.pob.Add(ind2)
            else:
                self.pob.Add(p1)
                self.pob.Add(p2)
            del worstPob[indx1]
            if indx1 > indx2:
                del worstPob[indx2]
            else:
                del worstPob[indx2-1]
        self.pob.poblation.sort(key=lambda x: x.fi,reverse=True)
    def Condition(self)->bool:
        return self.pob[0].fi == self.maxVolume
    def Seleccion(self,Pobl:list[Individuo],pt:float=0)->int:
        return self.Torneo(Pobl,pt)
    def Torneo(self,Pob:list[Individuo],pt:float=0.85)->int:
        n = len(Pob)
        i1 = random.randrange(0, n)
        i2 = random.randrange(0, n) 
        while i1 == i2:
            i2 = random.randrange(0, n)
        r = random.random()
        if r <= pt:
            if Pob[i1].fi > Pob[i2].fi:
                return i1
            else:
                return i2
        else:
            if Pob[i1].fi > Pob[i2].fi:
                return i2
            else:
                return i1