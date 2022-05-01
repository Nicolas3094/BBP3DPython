from Contenedor import Bin
import numpy as np
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
    cola = [[0,0,0]]
    for itemId in itemsToPack:
        for j,actualPoint in enumerate(cola):
            itemV = ITEMSDATA[itemId-1]
            if bin.Placement( PosiblePoint= np.array(actualPoint)+ np.array(itemV) ):
                if not bin.Overlap(pos=actualPoint, box =  itemV):
                    IterateDBLF(itemV,actualPoint,bin)
                    bin.addBox(id=itemId, pos = actualPoint, dimensions= itemV)
                    cola.pop(j)
                    for k in np.arange(3):
                        if actualPoint[k] + itemV[k] < bin.dimensions[k]:
                            actualPoint[k] += itemV[k]
                            cola.append(actualPoint.copy())
                            actualPoint[k] -= itemV[k]
                    bubbleSort(cola)
                    break        
    #path = "C:/Users/nicoo/My project/Assets/points.json"
    #PossiblePositions(path=path,positions=cola)

def DBLF2(bin:Bin, itemsToPack:list, ITEMSDATA:list):
    items = itemsToPack.copy()
    pointsQ = [[0,0,0]]
    while len(items) != 0 and len(pointsQ)!=0:
        point = pointsQ[0]
        pointsQ.pop(0)
        for i,itemId in enumerate(items):
            itemV = ITEMSDATA[itemId-1]
            if bin.Placement( PosiblePoint= np.array(point)+ np.array(itemV) ):
                if not bin.Overlap(pos=point, box =  itemV):
                    IterateDBLF(itemV,point,bin)
                    bin.addBox(id=itemId, pos = point, dimensions= itemV)
                    for k in np.arange(3):
                        if point[k] + itemV[k] < bin.dimensions[k]:
                            point[k] += itemV[k]
                            pointsQ.append(point.copy())
                            point[k] -= itemV[k]
                    bubbleSort(pointsQ)
                    items.pop(i)
                    break
    #path = "C:/Users/nicoo/My project/Assets/points.json"
    #PossiblePositions(path=path,positions=pointsQ)



    