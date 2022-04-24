from Contenedor import Bin
from DataLecture import PossiblePositions
import numpy as np
class ColaP(list):
    def __init__(self):
        self._visit=list()
        self=list()
    def push(self,punto:list):
        cad = str(punto[0])+str(punto[1])+str(punto[2])
        if(cad in self._visit):
            return
        self._visit.append(cad)
        n = self.size()
        self.append(punto)
    def __swap__(self,i,j):
        aux = self[i]
        self[i]=self[j]
        self[j]=aux
    def pop(self):
        if(self.empty()): return
        del self[0]
    def top(self)->list:
        return self[0]
    def size(self)->int:
        return len(self)
    def empty(self)->bool:
        return len(self)==0
    def Order(self):
        self.sort(key=lambda x:x[0])
        n = self.size()
        for i in range(n):
            for j in range(0, n-i-1):
                if( self[j+1][0]==self[j][0] and 
                   self[j+1][2] < self[j][2]):
                    self.__swap__(j,j+1)
                elif ( self[j+1][0]==self[j][0] and 
                    self[j+1][2] == self[j][2] and
                     self[j+1][1] < self[j][1]):
                    self.__swap__(j,j+1)

def IterateDBLF(item:list,actualPoint:list,bin:Bin):
    if bin.getN() <= 1 :
        return
    for _ in [0,2,1]:
        if actualPoint[_] == 0:
            continue
        while not bin.Overlap(pos=actualPoint, box =  item) and bin.Placement(PosiblePoint= np.array(actualPoint)+ np.array(item)):
            actualPoint[_]-=1
            if actualPoint[_]==-1:
                break
        actualPoint[_]+=1

def DBLF(bin:Bin, itemsToPack:list,ITEMSDATA:list):
    bin.reset()
    cola = ColaP()
    cola.push([0,0,0])    
    for itemId in itemsToPack:
        for j,actualPoint in enumerate(cola):
            itemV = ITEMSDATA[itemId-1]
            if bin.Placement( PosiblePoint= np.array(actualPoint)+ np.array(itemV) ):
                if not bin.Overlap(pos=actualPoint, box =  itemV):
                    IterateDBLF(itemV,actualPoint,bin)
                    bin.addBox(id=itemId, pos = actualPoint, dimensions= itemV)
                    del cola[j]
                    for k in range(3):
                        if actualPoint[k] + itemV[k] < bin.dimensions[k]:
                            actualPoint[k] += itemV[k]
                            cola.push(actualPoint.copy())
                            actualPoint[k] -= itemV[k]
                    cola.Order()
                    break        
    #path = "C:/Users/nicoo/My project/Assets/points.json"
    #PossiblePositions(path=path,positions=cola)

def DBLF2(bin:Bin, itemsToPack:list, ITEMSDATA:list):
    bin.reset()
    items = itemsToPack.copy()
    pointsQ = ColaP()
    pointsQ.push([0,0,0])
    while len(items) != 0 and not pointsQ.empty():
        point = pointsQ.top()
        pointsQ.pop()
        for i,itemId in enumerate(items):
            itemV = ITEMSDATA[itemId-1]
            if bin.Placement( PosiblePoint= np.array(point)+ np.array(itemV) ):
                if not bin.Overlap(pos=point, box =  itemV):
                    IterateDBLF(itemV,point,bin)
                    bin.addBox(id=itemId, pos = point, dimensions= itemV)
                    for k in range(3):
                        if point[k] + itemV[k] < bin.dimensions[k]:
                            point[k] += itemV[k]
                            pointsQ.push(point.copy())
                            point[k] -= itemV[k]
                    pointsQ.Order()
                    del items[i]
                    break
    #path = "C:/Users/nicoo/My project/Assets/points.json"
    #PossiblePositions(path=path,positions=pointsQ)



    