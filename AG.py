from Poblacion import Poblation,Individuo
from Contenedor import Bin
import numpy as np
import random
import copy
from Poblacion import Genome
from PackingH import DBLF
class AG:
    def __init__(self, pc:float, pt:float, pm:float):
        self._pc = pc #prob cruza
        self._pt = pt #prob de torneo
        self._pm = pm #prob de mutacion
        self._DataSet=dict()
        self.pob:Poblation = None

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
    
    def RandomInitPob(self,N:int , BoxSeq:list, Heuristic:bool=False):

        Pob = Poblation(N)

        originalInd = [i for i in range(1,len(BoxSeq)+1)]

        self._DataSet = dict(zip(originalInd,BoxSeq)) #Data set Global en forma de diccionario, de [1,..n] -> [ p1, p2, ..., pn  ], donde pi = [li wi hi]
        if Heuristic:
            N -= 4
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
        self._gen = maxGen
        for _ in np.arange(maxGen):
            self.EvaluateFitness(self.pob,contenedor,Data)
            self._RegistrarDatos__()
            if self.Condition():
                break
            self.CrearGeneracion()
        
    def EvaluateFitness(self,pobl:Poblation,dimBin,DataSet):
        for individuo in pobl:
            if individuo.fi is None:
                bin:Bin = Bin(dimensiones=dimBin, n=len(DataSet))
                self._packing(bin=bin,itemsToPack=individuo.genome, ITEMSDATA=DataSet)
                individuo.fi = bin.getLoadVol()/np.prod(dimBin)
        pobl.poblation.sort(key= lambda x : x.fi,reverse=True)

    def CrearGeneracion(self):
        n = random.randint(2,int(self.pob.n/2)) # maximo un cuarto de la mejor poblacion actual pasa a la siguiente , min 2 por elitismo, exento a cruza, los demas pasan a seleccion
        elite = 0
        worstPob:Poblation = Poblation(n)
        worstPob.poblation = self.pob.poblation[n:]
        nexPob:Poblation = Poblation(self.pob.n)
        nexPob.poblation = self.pob.poblation[:n]
        lim = self.pob.n-n
        while elite <  lim:
            if len(worstPob.poblation)==1:
                nexPob.Add(worstPob[0])
                break
            indx1 = self.Seleccion(worstPob,self._pt)
            indx2 = self.Seleccion(worstPob,self._pt)
            while indx1 == indx2:
                indx2 = self.Seleccion(worstPob,self._pt)
            p1:Individuo = worstPob[indx1]
            p2:Individuo = worstPob[indx2]
            rn = random.random()
            if rn <= self._pc:
                h1,h2 = self.Cruza(p1,p2)
                if elite < lim:
                    self.Mutation(h1,self._pm)
                    nexPob.Add(h1)
                    elite +=1
                if elite < lim:
                    self.Mutation(h2,self._pm)
                    nexPob.Add(h2)
                    elite +=1
                else:
                    break
            else:
                if elite < lim:
                    nexPob.Add(p1)
                    elite +=1
                if elite < lim:
                    nexPob.Add(p2)
                    elite +=1
                else:
                    break
            worstPob.DeleteInd(indx1)
            if indx1 > indx2:
                worstPob.DeleteInd(indx2)
            else:
                worstPob.DeleteInd(indx2-1)
        self.pob.poblation = nexPob.poblation

    def Condition(self):
        return self.pob[0].fi >=0.97

    def Seleccion(self,Pobl:Poblation,pt:float=0):
        return self.Torneo(Pobl,pt)
    
    def Torneo(self,Pob:Poblation,pt:float=0.85):
        n = len(Pob.poblation)
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
            
    def Mutation(self,ind:Individuo, PM:float):
        r = random.random()
        if r >= PM:
            return
        n = len(ind.genome)
        i= random.randrange(0,int(n/2))
        j= random.randrange(i+1,n)
        aux_code = ind.genome[0:i]+ind.genome[i:j][::-1]+ind.genome[j:n]
        ind.genome = list()
        ind.genome = aux_code
        
    def OX(self,P1:list,P2:list,i:int,j:int)->Genome:
        n = len(P1)
        h1 = [None]*n
        h1[i:j]=P1[i:j]
        for k in np.arange(j,len(P1)):
            for l in np.arange(0,len(P2)):
                if(P2[l] not in h1):
                    h1[k] = P2[l]
                    break
        for k in np.arange(0,i):
            for l in np.arange(0,len(P2)):
                if(P2[l] not in h1):
                    h1[k] = P2[l]
                    break 
        return h1
    
    def Cruza(self,P1:Individuo,P2:Individuo):

        n = len(P1.genome)
        i= random.randrange(3,int(n/2))
        j= random.randrange(i+1,n)

        h1 = Individuo(code=self.OX(P1.genome,P2.genome,i,j))

        h2 = Individuo(code=self.OX(P2.genome,P1.genome,i,j))
        
        return (h1,h2)