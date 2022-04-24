from Poblacion import Poblacion
from Poblacion import Individuo
import numpy as np
import random
import copy

class AG:
    def __init__(self, pc:float, pt:float, pm:float):
        
        self._pc = pc #prob cruza
        self._pt = pt #prob de torneo
        self._pm = pm #prob de mutacion
        self._DataSet=dict()

    def __initValues__(self):
        self._gen = 0 
        self.fiBest = list()
        self.fiWorst = list()
        self.Convergencia = list()

    def _RegistrarDatos__(self):
        self.fiBest.append(self.pob[0].fi)
        self.fiWorst.append(self.pob[-1].fi)
        self.Convergencia.append(self.pob.Convergence())

    def DefineProblem(self,specimen ):
        self.specimen = specimen

    def ConfigFi(self, evaluate):
        self._packingH = evaluate

    def Best(self):
        return self.pob[0]
    
    def RandomInitPob(self,N:int , BoxSeq:list, Heuristic:bool=False):

        Pob = Poblacion(N)

        originalInd = [i for i in range(1,len(BoxSeq)+1)]

        self._DataSet = dict(zip(originalInd,BoxSeq)) #Data set Global en forma de diccionario, de [1,..n] -> [ p1, p2, ..., pn  ], donde pi = [li wi hi]
        if Heuristic:
            N -= 4
            Pob.CreateInd(
                        BoxSeq = {k: v for k,v in sorted(self._DataSet.items(),key = lambda v: v[1][0]*v[1][1]*v[1][2])  },
                        param=copy.copy(self.specimen)
            ) #ordena por volumen
            for _ in range(3): #ordena por longitud, ancho y alto
                Pob.CreateInd(
                        BoxSeq = {k: v for k,v in sorted(self._DataSet.items(),key = lambda v: v[1][_])  },
                        param=copy.copy(self.specimen)
                )
        for _ in range(N):
            random.shuffle(originalInd)
            while originalInd in Pob:
                random.shuffle(originalInd)
            copyInd = originalInd.copy()
            Pob.CreateInd(code = copyInd, param=copy.copy(self.specimen)) 
        self.pob = Pob

        self.__initValues__()

    def train(self,maxGen):
        self._gen = maxGen
        g= 0
        self.EvaluateFitness(self.pob)

        for _ in range(maxGen):
            self._RegistrarDatos__()
            self.CrearGeneracion()
            self.EvaluateFitness(self.pob)

    def EvaluateFitness(self,poblation:Poblacion,DataSet):
        globalCajas = list(self._DataSet.values())
        for individuo in poblation:
            individuo.params.solve(individuo.code, globalCajas)
            individuo.fi = self._packingH(individuo.param)
        poblation.sort(key=lambda x: x.fi, reverse=True)  

    def CrearGeneracion(self):
        
        n = random.randint(2,int(self.pob.n/2)) # maximo un cuarto de la mejor poblacion actual pasa a la siguiente , min 2 por elitismo, exento a cruza, los demas pasan a seleccion
        elite = 0
        newPob = Poblacion(n)
        
        newPob = self.pob[n:]
        
        while elite <= n:
            indx1 = self.Seleccion(newPob,self._pt)
            indx2 = self.Seleccion(newPob,self._pt)
            while indx1 == indx2:
                indx2 = self.Seleccion(newPob,self._pt)
            p1 = newPob[indx1]
            p2 = newPob[indx2]
            rn = random.random()
            if rn <= self._pc:
                h1,h2 = self.Cruza(p1,p2)
                h1.params = self.specimen
                h2.params = self.specimen
                self.Mutation(h1,self._pm)
                self.Mutation(h2,self._pm)

                newPob.Add(h1)
                newPob.Add(h2)
            else:
                newPob.Add(p1)
                newPob.Add(p2)
            elite +=2
            del newPob[indx1]
            if indx1 > indx2:
                del newPob[indx2]
            else:
                del newPob[indx2-1]
        self.pob[n:]= newPob
    
    def Condition(self):
        return self.pob.poblacion[0].fi>=1

    def Seleccion(self,Pobl:Poblacion,pt:float=0):
        return self.Torneo(Pobl,pt)
    
    def Torneo(self,Pob:Poblacion,pt:float=0.85):
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
            
    def Mutation(self,ind:Individuo, PM:float):
        r = random.random()
        if r >= PM:
            return
        n = len(ind.code)
        i= random.randrange(0,int(n/2))
        j= random.randrange(i+1,n)
        aux_code = ind.code[0:i]+ind.code[i:j][::-1]+ind.code[j:n]
        ind.code = list()
        ind.code = aux_code
        
    def OX(self,P1:list,P2:list,i:int,j:int):
        n = len(P1)
        h1 = [None]*n
        h1[i:j]=P1[i:j]
        for k in range(j,len(P1)):
            for l in range(0,len(P2)):
                if(P2[l] not in h1):
                    h1[k] = P2[l]
                    break
        for k in range(0,i):
            for l in range(0,len(P2)):
                if(P2[l] not in h1):
                    h1[k] = P2[l]
                    break 
        return h1
    
    def Cruza(self,P1:Individuo,P2:Individuo):

        n = len(P1.code)
        i= random.randrange(0,int(n/2))
        j= random.randrange(i+1,n)

        h1 = Individuo()
        h1.code= self.OX(P1.code,P2.code,i,j)

        h2 = Individuo()
        h2.code = self.OX(P2.code,P1.code,i,j)
        
        return (h1,h2)