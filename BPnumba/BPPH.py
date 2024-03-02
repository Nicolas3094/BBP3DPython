import numpy as np
from numba import types, typed, njit, deferred_type
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from typing import List
from collections import OrderedDict

from BPnumba.BoxN import ItemBin
from BPnumba.PqueueNumba import CreatePriorityQueue, PQVector
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
specB['__rotBox'] = types.ListType(types.int64)
specB['__pos'] = types.ListType(types.ListType(types.int64))
specB['extrapts'] = types.ListType(types.ListType(types.int64))


@jitclass(specB)
class Bin:
    def __init__(self, dimensiones: List[int]):
        self.dimensions = dimensiones
        self.__loaded_volume = 0
        self.__n = 0
        self.__orderBox = typed.List.empty_list(types.int64)
        self.__rotBox = typed.List.empty_list(types.int64)
        self.__pos = typed.List.empty_list(typed.List.empty_list(types.int64))
        self.extrapts = typed.List.empty_list(
            typed.List.empty_list(types.int64))

    def addBox(self, box: ItemBin, itemPos: List[int]):
        self.__pos.append(itemPos)
        self.__orderBox.append(box.id)
        self.__rotBox.append(box.getRot())
        self.__loaded_volume += box.CDim()[0]*box.CDim()[1]*box.CDim()[2]
        self.__n += 1

    def getRot(self) -> List[int]:
        return self.__rotBox

    def getLoadVol(self) -> int:
        return self.__loaded_volume

    def getBoxes(self) -> List[int]:
        return self.__orderBox

    def getPositions(self) -> List[List[int]]:
        return self.__pos

    def getN(self):
        return self.__n


@njit
def create_Bin(dimensions):
    return Bin(dimensions)


Bin_type.define(Bin.class_type.instance_type)  # type: ignore


@njit
def Placement(Bin_Dim: List[int], PosiblePoint: List[int]) -> bool:
    return PosiblePoint[2] <= Bin_Dim[2] and PosiblePoint[1] <= Bin_Dim[1] and PosiblePoint[0] <= Bin_Dim[0]


@njit
def ABIntersect(Amax, Amin, Bmax, Bmin) -> bool:
    for i in np.arange(3):
        if Amin[i] >= Bmax[i] or Amax[i] <= Bmin[i]:
            return False
    return True


@njit
def Overlap(pos: List[int], currenBox: ItemBin, DataSet: List[ItemBin], Container: Bin, rot: int) -> bool:
    nwBox = currenBox.CDim()
    Amin = pos
    Amax = NumbaList([Amin[0]+nwBox[0], Amin[1]+nwBox[1], Amin[2]+nwBox[2]])
    for i in np.arange(Container.getN()):
        Bmin = NumbaList(Container.getPositions()[i])
        oldbox = DataSet[Container.getBoxes()[i]-1]
        oldbox.rotate(rotMode=Container.getRot()[i], rotWays=rot)
        oldbx = NumbaList(oldbox.CDim())
        Bmax = NumbaList(
            [Bmin[0] + oldbx[0], Bmin[1]+oldbx[1], Bmin[2]+oldbx[2]])
        if ABIntersect(Amax, Amin, Bmax, Bmin):
            return True
    return False



@njit
def BetterCorner(pt:List[int],bin:Bin,BoxesData:List[ItemBin],wayRotation:int):
    indices = []
    Xindices = []
    Yindices = []
    tmpZ = pt[2]
    idx = []
    idz = []
    idy = []

    y0 = pt[1]
    x0 = pt[0]
    zm = 0
    m = 0
    
    #Primero sobre Z
    for v in np.arange(bin.getN()):

        bx = BoxesData[bin.getBoxes()[v]-1]
        bx.rotate(bin.getRot()[v], wayRotation)
        zi = bx.CDim()[2]
        #print("pz ",pt[2])
        #print(bx.id,zm, bin .getPositions()[v], zi)
        if zm <= bin.getPositions()[v][2] + zi and bin.getPositions()[v][2] + zi <= pt[2]:
            zm = bin.getPositions()[v][2] + zi
            m += 1
            indices.append(bx.id)
            idx.append(v)
            
    #print("indices: ",indices)
    if len(indices)==0:
        return
    # Sobre x
    for k in np.arange(m):
        bx = BoxesData[indices[k]-1]
        bx.rotate(bin.getRot()[idx[k]], wayRotation)
        Di = bx.CDim()[0]
        if x0 <= Di + bin.getPositions()[idx[k]][0] and x0 >= bin.getPositions()[idx[k]][0]:
            Xindices.append(bx.id)
            idz.append(idx[k])
    if len(Xindices) == 0:
        return
    #print("Xindices: ",Xindices)
    # sobre y
    for k in np.arange(len(Xindices)):
        bx = BoxesData[Xindices[k]-1]
        bx.rotate(bin.getRot()[idz[k]], wayRotation)
        Wi = bx.CDim()[1]
        if y0 >= bin.getPositions()[idx[k]][1] and Wi+bin.getPositions()[idx[k]][1] >= y0 :
            Yindices.append(bx.id)
            idy.append(idz[k])
    #print("Yindices " ,Yindices )
    if len(Yindices) != 0:
        bx = BoxesData[Yindices[-1]-1]
        bx.rotate(bin.getRot()[idy[-1]], wayRotation)
        zp = bx.CDim()[2] + bin.getPositions()[idy[-1]][2]
        pt[2] = zp
        #print(bx.CDim(), " id",idy[-1])
        #print("cambia ",pt,zp)

@njit
def IterateDBLF(pos: List[int], box: ItemBin, DataSet: List[ItemBin], contianer: Bin, rot: int):
    if contianer.getN() <= 1:
        return
    for _ in [0, 2,1]:
        if pos[_] == 0:
            continue
        while not Overlap(pos=pos, currenBox=box, DataSet=DataSet, Container=contianer, rot=rot):
            pos[_] -= 1
            if pos[_] == -1:
                break
        pos[_] += 1


@njit
def AddBox(lstP: PQVector, bin: Bin, pt: List[int], boxID: ItemBin,itemsRor:int,genome:List[int] ,itemV: List[int], BoxesData: List[ItemBin], rot: int):
    IterateDBLF(pos=pt,box= boxID,DataSet= BoxesData,contianer=bin,rot=rot)
    boxID.rotate(itemsRor, rot)

    bin.addBox(boxID, pt.copy())
    #crns = CORNERS3D(cont=bin,BoxesData=BoxesData,rotItems=itemsRor,itemsToPack=genome,rot=rot)
    #if bin.getN()==1:
    
    for k in np.arange(3):
            if pt[k] + itemV[k] < bin.dimensions[k]:
                pt[k] += itemV[k]
                ptx=pt.copy()
                #BetterCorner(pt=ptx,bin=bin,BoxesData=BoxesData,wayRotation=rot)
                lstP.push(NumbaList(ptx))
                #lstP.push(NumbaList(pt))
                pt[k] -= itemV[k]
    #    return
    #CORNERS3D(CornerPoints=lstP ,cont=bin,BoxesData=BoxesData,rotItems=itemsRor,itemsToPack=genome,rot=rot)
    lstP.updateList()

    
@njit
def DBLF(bin: Bin, itemsToPack: List[int], itemsRor: list[int], BoxesData: List[ItemBin], wayRotation: int = 0):
    lstP = CreatePriorityQueue(NumbaList([0, 2, 1]))  # Define el orde x-z-y
    lstP.push(NumbaList([0,0,0]))
    for i in np.arange(len(itemsToPack)):  # La codificacion y BoxData deben
        boxID = itemsToPack[i]
        box = BoxesData[boxID-1]
        box.rotate(itemsRor[i], wayRotation)
        if not box.isValidRot():
            continue
        for j in np.arange(lstP.size()):
            pt = lstP.getPt(j)
            if Placement(bin.dimensions, NumbaList([pt[0]+box.CDim()[0], pt[1]+box.CDim()[1], pt[2]+box.CDim()[2]])):
                overlap = Overlap(
                    pos=pt, currenBox=box, DataSet=BoxesData, Container=bin, rot=wayRotation)
                if not overlap:
                    lstP.delPt(j)
                    box.rotate(itemsRor[i], wayRotation)
                    AddBox(lstP=lstP, bin=bin, pt=pt, boxID=box,genome=itemsToPack,itemsRor=itemsRor[i],
                           itemV=box.CDim(), BoxesData=BoxesData, rot=wayRotation)       
                    break
                else:          
                    box.rotate(itemsRor[i], wayRotation)

@njit
def CORNERS2D(cont:Bin,BoxesData:List[ItemBin],I:List[int],rotItems:List[int],itemsToPack:List[int],rot:int):
    if len(I)==0:
        return NumbaList([ NumbaList(np.array([0,0],dtype=np.int64))])
    #Identificar puntos extremos
    em=[]
    m=0
    ym=0
    for j in np.arange(len(I)):
        box =  BoxesData[cont.getBoxes()[I[j]]-1]
        box.rotate(cont.getRot()[I[j]], rot)
        wj = box.CDim()[1]
        yj = cont.getPositions()[I[j]][1]
        
        if yj + wj > ym:
            m +=1
            em.append(I[j])
            ym=wj+yj
    
    
    #Determinar puntos esquina
    box_e1= BoxesData[cont.getBoxes()[em[0]]-1]
    box_e1.rotate(cont.getRot()[em[0]], rot)
    
    ze1=cont.getPositions()[em[0]][2]
    he1= box_e1.CDim()[2]

    C= [[0,ze1+he1]]
    for j in np.arange(2,m):
        box_e=BoxesData[cont.getBoxes()[em[j]]-1]
        box_e.rotate(cont.getRot()[em[j]], rot)
        
        he=box_e.CDim()[2]
        ze = cont.getPositions()[em[j]][2]
        
        
        box_pe=BoxesData[cont.getBoxes()[em[j-1]]-1]
        box_pe.rotate(cont.getRot()[em[j-1]], rot)
        
        wpe=box_e.CDim()[1]
        ype = cont.getPositions()[em[j-1]][1]
        
        C.append([ype+wpe,ze+he])
        
    box_m=BoxesData[cont.getBoxes()[em[-1]]-1]
    box_m.rotate(cont.getRot()[em[-1]], rot)
    wm=box_m.CDim()[1]
    ym = cont.getPositions()[em[-1]][1]
    C.append([ym+wm,0])
    w_min=99999999
    h_min=99999999
    for p in np.arange(cont.getN(),len(itemsToPack)):
        bx=BoxesData[itemsToPack[p]-1]
        bx.rotate(rotItems[p], rot)
        wp=bx.CDim()[1]
        hp=bx.CDim()[2]
        if w_min>wp:
            w_min=wp
        if h_min > hp:
            h_min=hp
    newCorners:list[list[int]] = []
    for pt  in C:
        if not (pt[0]+w_min > cont.dimensions[1] or pt[1]+h_min>cont.dimensions[2]):
            newCorners.append(NumbaList(np.array(pt,dtype=np.int64)))
    newCorners=NumbaList(newCorners)
    return newCorners

    
@njit
def CORNERS3D(CornerPoints :PQVector,cont:Bin,BoxesData:List[ItemBin],rotItems:List[int],itemsToPack:List[int],rot:int):
    if cont.getN()==0:
        CornerPoints.push(  NumbaList([0,0,0]))
        return
    T=[0]
    
    for j in np.arange(cont.getN()):
        box =  BoxesData[cont.getBoxes()[j]-1]
        box.rotate(cont.getRot()[j], rot)
        
        dj = box.CDim()[0]
        xj = cont.getPositions()[j][0]
        
        if xj + dj not in T:
            T.append(xj + dj)

    T.sort()
    kp=1
    r=len(T)

    d_min=99999999
    for p in np.arange(cont.getN(),len(itemsToPack)):
        bx=BoxesData[itemsToPack[p]-1]
        bx.rotate(rotItems[p], rot)
        dp=bx.CDim()[0]
        if d_min>dp:
            d_min=dp
    if d_min == 99999999:
        return

    for k in np.arange(1,r):
        if  T[k]+d_min>cont.dimensions[0]:
            break
        Ik=[]
        for i in np.arange(cont.getN()):
            box =  BoxesData[cont.getBoxes()[i]-1]
            box.rotate(cont.getRot()[i], rot)
            di=box.CDim()[0]
            xi = cont.getPositions()[i][0]
            if xi+di>T[k]:
                Ik.append(i)
        Ikprev=[]
        for i in np.arange(cont.getN()):
            box =  BoxesData[cont.getBoxes()[i]-1]
            box.rotate(cont.getRot()[i], rot)
            di=box.CDim()[0]
            xi = cont.getPositions()[i][0]
            if xi+di>T[k-1]:
                Ikprev.append(i)
        Ck=CORNERS2D(cont=cont,BoxesData=BoxesData,I=Ik,rotItems=rotItems,itemsToPack=itemsToPack,rot=rot)
        Ckprev=CORNERS2D(cont=cont,BoxesData=BoxesData,I=Ikprev,rotItems=rotItems,itemsToPack=itemsToPack,rot=rot)
        for s in np.arange(len(Ck)):
            if Ck[s] not in Ckprev:
                CornerPoints.push(NumbaList([T[k],Ck[s][0],Ck[s][1]]))
