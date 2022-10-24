import re
from sys import ps1
import numpy as np
from numba import types, typed, njit,deferred_type
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from typing import List
from collections import OrderedDict

from regex import B
from BPnumba.BoxN import ItemBin
from BPnumba.PqueueNumba import CreatePriorityQueue,PQVector
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


Bin_type = deferred_type()
specB = OrderedDict()
specB['dimensions'] = types.ListType(types.int64)
specB['__loaded_volume'] = types.int64
specB['__n'] = types.int64
specB['__orderBox'] = types.ListType(types.int64)
specB['__pos'] =  types.ListType( types.ListType(types.int64))
@jitclass(specB)
class Bin:
    def __init__(self, dimensiones:List[int]):
        self.dimensions = dimensiones
        self.__loaded_volume = 0
        self.__n=0
        self.__orderBox = typed.List.empty_list(types.int64)
        self.__pos = typed.List.empty_list(typed.List.empty_list(types.int64))
    
    def addBox(self,box:ItemBin ,itemPos:List[int]):
        self.__pos.append(itemPos)
        self.__orderBox.append(box.id)
        self.__loaded_volume += box.CDim()[0]*box.CDim()[1]*box.CDim()[2]
        self.__n+=1

    def getLoadVol(self)->int:
        return self.__loaded_volume
    def getBoxes(self)->List[int]:
        return self.__orderBox
    def getPositions(self)->List[List[int]]:
        return self.__pos
    def getN(self):
        return self.__n
@njit
def create_Bin(dimensions):
    return Bin(dimensions)
Bin_type.define(Bin.class_type.instance_type)

@njit
def Placement(Bin_Dim:List[int],PosiblePoint:List[int])->bool:
        return PosiblePoint[2] <= Bin_Dim[2] and PosiblePoint[1] <=  Bin_Dim[1] and PosiblePoint[0] <= Bin_Dim[0] 
@njit
def ABIntersect(Amax,Amin,Bmax,Bmin)->bool:
    for i in np.arange(3):
        if Amin[i] >= Bmax[i] or Amax[i] <= Bmin[i]:
            return False
    return True
@njit 
def Overlap(pos:List[int],currenBox:ItemBin,DataSet:List[ItemBin],orderBox:List[int],positionBox:List[List[int]] )->bool:
        nwBox = currenBox.CDim()
        Amin = pos
        Amax = NumbaList([Amin[0]+nwBox[0],Amin[1]+nwBox[1],Amin[2]+nwBox[2]])
        for i in np.arange(len(orderBox)):
                Bmin =  NumbaList(positionBox[i])
                oldbx =NumbaList(DataSet[orderBox[i]-1].CDim())
                Bmax = NumbaList([Bmin[0] +oldbx[0],Bmin[1]+oldbx[1],Bmin[2]+oldbx[2]])
                if ABIntersect(Amax,Amin,Bmax,Bmin):
                    return True
        return False
@njit 
def IterateDBLF(pos:List[int], box:ItemBin, DataSet:List[ItemBin],orderBox:List[int],posBoxes:List[List[int]] ):
    if len(orderBox) <= 1 :
        return
    for _ in [0,2,1]:
        if pos[_] == 0:
            continue
        while not Overlap(pos = pos,currenBox= box,DataSet = DataSet,orderBox= orderBox, positionBox= posBoxes ):
            pos[_]-=1
            if pos[_]==-1:
                break
        pos[_]+=1

@njit 
def AddBox(lstP:PQVector,bin:Bin,pt:List[int],boxID:ItemBin,itemV:List[int],BoxesData:List[ItemBin]):
    IterateDBLF(pos=pt,box= boxID,DataSet= BoxesData,orderBox= bin.getBoxes(), posBoxes= bin.getPositions())
    bin.addBox(boxID,pt)
    for k in np.arange(3):
        if pt[k] + itemV[k] < bin.dimensions[k]:
           pt[k] += itemV[k]
           lstP.push(NumbaList([pt[0],pt[1],pt[2]]))
           pt[k] -= itemV[k]
    lstP.updateList()    

@njit 
def DBLF2(bin:Bin, itemsToPack:List[int], DataSet:List[List[int]])->None:
    boxes = NumbaList(itemsToPack.copy())
    lstP = CreatePriorityQueue(NumbaList([0,2,1]))
    lstP.push(NumbaList([0,0,0]))
    while len(boxes) != 0 and not lstP.empty():
        pt = lstP.top()
        lstP.pop() 
        for i in np.arange(len(boxes)):
            boxID = boxes[i]
            itemV = DataSet[boxID-1]
            if Placement(bin.dimensions, NumbaList([pt[0]+itemV[0],pt[1]+itemV[1],pt[2]+itemV[2]])):
                overlap = Overlap(pt,boxID,DataSet,bin.getBoxes(),bin.getPositions())
                if not overlap:
                    boxes.pop(i)
                    AddBox(lstP,bin,pt,boxID,itemV,DataSet)
                    break  

#El conjunto de las cajas "BoxData" tiene el mismo ORDEN que el e "itemsToPackl"
@njit
def DBLF(bin:Bin, itemsToPack:List[int],itemsRor:list[int],BoxesData:List[ItemBin], wayRotation:int=0):
    lstP = CreatePriorityQueue(NumbaList([0,2,1])) #Define el orde x-z-y
    
    lstP.push(NumbaList([0,0,0])) #Inicia con el origen
    
    for i in np.arange(len(itemsToPack)): #La codificacion y BoxData deben
        
        boxID = itemsToPack[i]
        box =  BoxesData[boxID-1]
        if wayRotation!=0:
            box.rotate(itemsRor[i],wayRotation)
        if not box.isValidRot(): 
            continue
        
        for j in np.arange(lstP.size()):
            pt = lstP.getPt(j)
            if  Placement(bin.dimensions,NumbaList([pt[0]+box.CDim()[0],pt[1]+box.CDim()[1],pt[2]+box.CDim()[2]])):
                overlap = Overlap(pos = pt,currenBox= box,DataSet= BoxesData,orderBox= bin.getBoxes(),positionBox= bin.getPositions())
                if not overlap:
                    lstP.delPt(j)
                    AddBox(lstP = lstP,bin =  bin,pt= pt,boxID= box,itemV= box.CDim(),BoxesData= BoxesData)           
                    break
  