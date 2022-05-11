import numpy as np
from numba import types, typed, njit,deferred_type
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from typing import List
from collections import OrderedDict
from BPnumba.PqueueNumba import PQVector
import random
@njit
def CreatePermutation(ls1:List[int]):
    repeated = []
    while len(repeated)!=len(ls1):
        randNum = random.randint(1,len(ls1))
        if randNum not in repeated:
            repeated.append(randNum)
    return repeated
@njit
def CreatePoblation(num:int, ls2:List[int])->List[List[int]]:
    poblation = []
    for _ in np.arange(num):
        pi =  CreatePermutation(ls2)
        if pi not in poblation:
            poblation.append(pi)
    return poblation


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

    def addBox(self, itemID:int, itemPos:List[int], itemDim:List[int]):
        self.__pos.append(itemPos)
        self.__orderBox.append(itemID)
        self.__loaded_volume += itemDim[0]*itemDim[1]*itemDim[2]
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

@njit#(boolean(int64[:],int64[:]),nogil=True)
def Placement(Bin_Dim:List[int],PosiblePoint:List[int])->bool:
        return PosiblePoint[2] <= Bin_Dim[2] and PosiblePoint[1] <=  Bin_Dim[1] and PosiblePoint[0] <= Bin_Dim[0] 
@njit
def ABIntersect(Amax,Amin,Bmax,Bmin)->bool:
    for i in range(3):
        if Amin[i] >= Bmax[i] or Amax[i] <= Bmin[i]:
            return False
    return True
@njit#(boolean(int64[:],int64,int64[:,:],int64[:],int64[:,:]),nogil=True)
def Overlap(pos:List[int],boxId:int,DataSet:List[List[int]],orderBox:List[int],positionBox:List[List[int]] )->bool:
        boxId -=1
        nwBox = NumbaList(DataSet[boxId])
        Amin = pos
        Amax = NumbaList([Amin[0]+nwBox[0],Amin[1]+nwBox[1],Amin[2]+nwBox[2]])
        for i in np.arange(len(orderBox)):
                Bmin =  NumbaList(positionBox[i])
                oldbx =NumbaList(DataSet[orderBox[i]-1])
                Bmax = NumbaList([Bmin[0] +oldbx[0],Bmin[1]+oldbx[1],Bmin[2]+oldbx[2]])
                if ABIntersect(Amax,Amin,Bmax,Bmin):
                        return True
        return False
@njit#(void(int64[:],int64,int64[:,:],int64[:],int64[:,:]),nogil=True)
def IterateDBLF(pos:List[int], boxId:int,DataSet:List[List[int]],orderBox:List[int],posBoxes:List[List[int]] ):
    if len(orderBox) <= 1 :
        return
    for _ in [0,2,1]:
        if pos[_] == 0:
            continue
        while not Overlap(pos,boxId,DataSet,orderBox,posBoxes ):
            pos[_]-=1
            if pos[_]==-1:
                break
        pos[_]+=1

@njit #(void(Bin_type, int64[:],int64[:,:]),nogil=True)
def NumDBLF(bin:Bin, itemsToPack:List[int], DataSet:List[List[int]])->None:
    boxes = NumbaList(itemsToPack.copy())
    lstP = PQVector(NumbaList([0,2,1]))
    lstP.push(NumbaList([0,0,0]))
    while len(boxes) != 0 and not lstP.empty():
        pt = lstP.top()
        for i in np.arange(len(boxes)):
            boxID = boxes[i]
            itemV = DataSet[boxID-1]
            posiblePt = NumbaList([pt[0]+itemV[0],pt[1]+itemV[1],pt[2]+itemV[2]])
            if Placement(bin.dimensions, posiblePt):
                NoOverlap = not Overlap(pt,boxID,DataSet,bin.getBoxes(),bin.getPositions())
                if NoOverlap:
                    IterateDBLF(pt,boxID,DataSet,bin.getBoxes(),bin.getPositions())
                    bin.addBox(boxID,pt,itemV)
                    for k in np.arange(3):
                        if pt[k] + itemV[k] < bin.dimensions[k]:
                            pt[k] += itemV[k]
                            newpt = pt.copy()
                            lstP.push(NumbaList(newpt))
                            pt[k] -= itemV[k]
                    boxes.pop(i)
                    break
        if (lstP.size() == 1):
            return
        lstP.pop() #Hay error aqui por Numba me pide que retorne el tipo de dato de Nodo de la lista de proridad, chale   


