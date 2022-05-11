from Contenedor import Bin
import numpy as np
from Pqueue import PQVector
def bubbleSort(arr:list[list[int]])->None:
    n = len(arr)
    for j in np.arange(n-1):
        for i in np.arange(0,n-j-1):
            cond1 = arr[i][0] > arr[i+1][0]
            cond2 = arr[i][0] == arr[i+1][0] and arr[i][2] > arr[i+1][2]
            cond3 = arr[i][0] == arr[i+1][0] and arr[i][2] == arr[i+1][2] and arr[i][1] > arr[i+1][1]
            cond4 = arr[i][0] == arr[i+1][0] and arr[i][2] == arr[i+1][2] and arr[i][1] == arr[i+1][1]
            if cond1 or cond2 or cond3:
                arr[i],arr[i+1]=arr[i+1],arr[i]

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
    points = [[0,0,0]]
    visited = []
    for itemId in itemsToPack:
        for j in range(len(points)):
            itemV = ITEMSDATA[itemId-1]
            actualPoint = points[j]
            if bin.Placement( PosiblePoint= np.array(actualPoint)+ np.array(itemV) ):
                if not bin.Overlap(pos=actualPoint, box =  itemV):
                    IterateDBLF(itemV,actualPoint,bin)
                    bin.addBox(id=itemId, pos = actualPoint, dimensions= itemV)
                    points.pop(j)
                    for k in np.arange(3):
                        if actualPoint[k] + itemV[k] < bin.dimensions[k]:
                            actualPoint[k] += itemV[k]
                            if actualPoint not in visited:
                                points.append(actualPoint.copy())
                            else:
                                visited.append(actualPoint)
                            actualPoint[k] -= itemV[k]
                    bubbleSort(points)
                    break        
    #path = "C:/Users/nicoo/My project/Assets/points.json"
    #PossiblePositions(path=path,positions=cola)

def DBLF2(bin:Bin, itemsToPack:list, ITEMSDATA:list):
    items = itemsToPack.copy()
    pointsQ = PQVector([0,2,1]) #orden x,z,y 
    pointsQ.push([0,0,0])
    visited = []
    while len(items) != 0 and not pointsQ.empty():
        point = pointsQ.pop()
        for i,itemId in enumerate(items):
            itemV = ITEMSDATA[itemId-1]
            if bin.Placement( PosiblePoint= np.array(point)+ np.array(itemV) ):
                if not bin.Overlap(pos=point, box =  itemV):
                    IterateDBLF(itemV,point,bin)
                    bin.addBox(id=itemId, pos = point, dimensions= itemV)
                    for k in np.arange(3):
                        if point[k] + itemV[k] < bin.dimensions[k]:
                            point[k] += itemV[k]
                            if point not in visited:
                                pointsQ.push(point.copy())
                            else:
                                visited.append(point)
                            point[k] -= itemV[k]
                    items.pop(i)
                    break
    #path = "C:/Users/nicoo/My project/Assets/points.json"
    #PossiblePositions(path=path,positions=pointsQ)



    