import numpy as np

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
        self.ni = [ 0 for i in range(n)]
        self._loaded_volume = 0
        self._prevBox:Item = None
        self._n=0

    def addBox(self, id:int, pos:list, dimensions:list):
        currentItem = Item(classType=id, minMaxVertex=[pos, np.array(pos)+np.array(dimensions)])
        self._loadedItems.append(currentItem)
        self._loaded_volume += np.prod(dimensions)
        self.ni[id-1] = 1 
        self._prevBox = currentItem
        self._n+=1

    def getLoadVol(self)->float:
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