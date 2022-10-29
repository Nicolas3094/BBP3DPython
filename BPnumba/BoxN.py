import numpy as np
from numba import types, njit,deferred_type
from numba.experimental import jitclass
from collections import OrderedDict
from numba.typed import List as NumbaList

#Las dimensiones van de:
#               x->Largo (L)
#               y->Ancho (W)
#               z->Alto (H)

Box_type =deferred_type()
sepecBox = OrderedDict()


sepecBox['id'] = types.int64
sepecBox['T'] = types.int64
sepecBox['__rot__'] = types.int64

sepecBox['__wi__'] = types.int64 #0
sepecBox['__hi__'] = types.int64 #1
sepecBox['__li__'] = types.int64 #2

sepecBox['dimension'] = types.ListType(types.int64) #toma valores de [0,1,2], representa las dimensiones de la caja
sepecBox['position'] = types.ListType(types.int64)
sepecBox['n'] = types.int64
sepecBox['__rotation__'] = types.ListType(types.int64)
@jitclass(sepecBox)
class ItemBin:
    def __init__(self, boxData:list[list[int]],id :int,tipo:int):

        self.id = id
        self.T=tipo
        self.__rot__=0
        self.__li__=boxData[0][0] #0
        self.__wi__=boxData[1][0] #1
        self.__hi__=boxData[2][0] #2
        self.n = 0
        self.__rotation__ =  NumbaList(np.zeros(3,dtype=np.int64))
        self.__rotation__[0]=boxData[0][1]
        self.__rotation__[1]=boxData[1][1]
        self.__rotation__[2]=boxData[2][1]
        self.dimension = NumbaList(np.arange(3,dtype=np.int64))        
        self.position =  NumbaList(np.ones(3,dtype=np.int64)*-1)
        for i in np.arange(3):
            self.__rotation__[i] = boxData[i][1]

    def RDim(self)->list[int]:
        return NumbaList(np.array([self.__li__,self.__wi__,self.__hi__],dtype=np.int64) )

    def CDim(self)->list[int]:
        curren_dim=  NumbaList(np.zeros(3,dtype=np.int64))
        for i in np.arange(3):
            if self.dimension[i]==0:
                curren_dim[i]=self.__li__
            elif self.dimension[i]==1:
                curren_dim[i]=self.__wi__
            else:
                curren_dim[i]=self.__hi__
        return curren_dim

    def getRot(self)->int:
        return self.__rot__

    #Solo rotaciones donde la posicon vertical (z) sea valida    
    def isValidRot(self):
        return self.__rotation__[self.dimension[2]] == 1
        
    def rotate(self, rotMode:int, rotWays:int=6)->None:
        if( (rotMode < 0 or rotMode>5) and rotWays == 6 ) or ( (rotMode < 0 or rotMode>1) and rotWays == 2 ) or (rotWays != 6 and rotWays !=2):
            return
        self.__rot__=rotMode
        if rotMode==0:
            self.dimension=NumbaList([0,1,2])
        elif rotMode==1 and rotWays == 6:
            self.dimension=NumbaList([0,2,1])
        elif (rotMode==2 and rotWays == 6) or (rotMode==1 and rotWays==2):
            self.dimension=NumbaList([1,0,2]) # 2 way rotatopm
        elif rotMode==3 and rotWays == 6:
            self.dimension=NumbaList([1,2,0])
        elif rotMode==4 and rotWays == 6:
            self.dimension=NumbaList([2,0,1])
        else:
            self.dimension=NumbaList([2,1,0])
Box_type.define(ItemBin.class_type.instance_type)  # type: ignore
@njit
def create_ItemBin(BoxData:list[list[int]],typ:int,i :int)->ItemBin:
    return ItemBin(NumbaList(BoxData),i,typ)

@njit
def InstaceBoxes(Data)->list[ItemBin]:
    boxes=[]     
    types=0
    count=0
    for i in np.arange(len(Data)):
        
        if i==0:
            prevboxData=Data[0]
            types=1
        else:
            prevboxData=Data[i-1]
        if   prevboxData != Data[i]:
            types +=1
            count=1
        else:
            count +=1 
        bpx = create_ItemBin(BoxData=Data[i],i=i+1,typ=types)
        boxes.append(bpx)    
    return NumbaList(boxes)