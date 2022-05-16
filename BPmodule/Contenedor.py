import numpy as np
import random 

class Item:
    def __init__(self,classType=0,dimensions=None, minMaxVertex = None):
        self.id = classType
        self.dimensions = np.array(dimensions)
        if minMaxVertex is not None:
            self.MinV = np.array(minMaxVertex[0])
            self.MaxV = np.array(minMaxVertex[1])
            self.dimensions = np.abs(self.MaxV-self.MinV)
        else:
             self.MinV =None
             self.MaxV=None
        self.Vol = np.prod(self.dimensions)
    def AddPoint(self, point:list):
        self.MinV = np.array(point)
        self.MaxV = self.dimensions + self.MinV

class Bin:
    def __init__(self, dimensiones:list, n:int):
        self._loadedItems = list()
        self.dimensions = dimensiones
        self._loaded_volume = 0
        self._n=0
    def addBox(self, id:int, pos:list, dimensions:list):
        currentItem = Item(classType=id, minMaxVertex=[pos, np.array(pos)+np.array(dimensions)])
        self._loadedItems.append(currentItem)
        self._loaded_volume += np.prod(dimensions)
        self._n+=1
    def getLoadVol(self)->int:
        return self._loaded_volume
    def getBox(self, index:int)->Item:
        return self._loadedItems[index]
    def getBoxes(self)->list:
        return self._loadedItems
    def reset(self):
        self._n=0
        self._loaded_volume = 0
        self._loadedItems.clear()
    def getN(self):
        return self._n
    def Placement(self, PosiblePoint:list)->bool:
        return PosiblePoint[2] <= self.dimensions[2] and PosiblePoint[1] <=  self.dimensions[1] and PosiblePoint[0] <= self.dimensions[0] 
    def Overlap(self,pos:list, box:list )->bool:
        Amin = np.array(pos)
        Amax = Amin + np.array(box)
        for loadItem in self._loadedItems:
            Bmin = loadItem.MinV
            Bmax = loadItem.MaxV
            if self.ABIntersect(Amax,Amin,Bmax,Bmin):
                return True
        return False
    def ABIntersect(self, Amax,Amin,Bmax,Bmin)->bool:
        for i in range(3):
            if Amin[i] >= Bmax[i] or Amax[i] <= Bmin[i]:
                return False
        return True

def RandomPoblation(N:int , BoxSeq:list, Heuristic:bool=False)->list[list[int]]:
        poblation = list()
        originalInd = [i for i in range(1,len(BoxSeq)+1)]
        _DataSet = dict(zip(originalInd,BoxSeq)) #Data set Global en forma de diccionario, de [1,..n] -> [ p1, p2, ..., pn  ], donde pi = [li wi hi]
        if Heuristic:
            N -= 4
            vol = list({k: v for k,v in sorted(_DataSet, reverse=True,key = lambda v: v[1][0]*v[1][1]*v[1][2])}.keys())
            poblation.append(vol)
            for _ in range(3): #ordena por longitud, ancho y alto
                di = list({k: v for k,v in sorted(_DataSet.items(),reverse=True,key = lambda v: v[1][_])}.keys())
                poblation.append(di)
        for _ in range(N):
            random.shuffle(originalInd)
            while originalInd in poblation:
                random.shuffle(originalInd)
            copyInd = originalInd.copy()
            poblation.append(copyInd)
        return poblation