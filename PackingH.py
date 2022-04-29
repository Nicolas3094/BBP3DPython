from Contenedor import Bin
import numpy as np
def quickSort(arr,low,high):
    def partition(arr,low,high):
        i = (low-1)
        pivot = arr[high]
        for j in np.arange(low,high):
            cond3 =  arr[j][0] == pivot[0] and arr[j][2] == pivot[2] and arr[j][1] < pivot[1]
            cond1 = arr[j][0] == pivot[0] and arr[j][2] < pivot[2]
            cond2 = arr[j][0] < pivot[0]
            if cond1 or cond2 or cond3:
                i = i+1
                arr[i], arr[j] = arr[j],arr[i]
        arr[i-1], arr[high] = arr[high],arr[i-1]
        return (i+1)
    if len(arr) == 1:
        return arr
    if low < high:
        pi = partition(arr,low,high)
        quickSort(arr,low,pi-1)
        quickSort(arr,pi+1,high)

class ColaP(list):
    def __init__(self):
        self=list()
    def push(self,punto:list):
        self.append(punto)
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
        quickSort(self,0,len(self)-1)

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



    