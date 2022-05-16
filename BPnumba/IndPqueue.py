from cmath import pi
from numba import types,njit,deferred_type, optional
from numba.experimental import jitclass
from collections import OrderedDict

from sqlalchemy import null
from sympy import Q, preview
from BPnumba.NumAG import ind_type,Ind
import numpy as np
from numba.typed import List as NumbaList


nodeInd_type = deferred_type()
specNode = OrderedDict()
specNode['data'] = optional(ind_type)
specNode['Rnode'] = optional(nodeInd_type)
specNode['Lnode'] = optional(nodeInd_type)
@jitclass(specNode)
class GNodo(object):
    def __init__(self,data:ind_type,Right:nodeInd_type,Left:nodeInd_type):
        self.data = data
        self.Rnode = Right
        self.Lnode = Left
@njit
def make_linked_Gnode(data:optional(ind_type),right:optional(nodeInd_type),left:optional(nodeInd_type)):
    return GNodo(data,right,left)
nodeInd_type.define(GNodo.class_type.instance_type)

listPQ_type = deferred_type()
specPQ = OrderedDict()
specPQ['__n'] = types.int64
specPQ['__top'] = optional(nodeInd_type)
specPQ['__back'] = optional(nodeInd_type)
specPQ['__maxN'] = types.int64
specPQ['__maxFi'] = types.float64
specPQ['__minFi'] = types.float64

@jitclass(specPQ)
class GPQueue:
    def __init__(self,fixedNum:int):
        self.__n = 0
        self.__maxN = fixedNum
        self.__maxFi = 0
        self.__minFi = 1
        self.__top =None
        self.__back = None
        
    def push(self, ind:Ind )->None:
        if self.__back is None:
            self.__top= make_linked_Gnode(ind,None,None)
            self.__back =self.__top
        else:
            self.__back.Rnode=make_linked_Gnode(ind,None,None)
            self.__back.Rnode.Lnode=self.__back
            self.__back = self.__back.Rnode
        self.__n+=1
        
    def Print(self):
        top = self.__top
        while top is not None:
            print(top.data.fi,top.data.genome)
            top = top.Rnode
    def GetMaxFi(self):
        return self.__maxFi
    def GetMinFi(self):
        return self.__minFi
    def pop(self):
        if not self.empty():
            tmp = self.__top
            if self.size()==1:
                self.__n=0
                return
            self.__top = tmp.Rnode
            self.__n-=1
    def popBack(self):
        self[self.__n-2].Rnode=None
        self.__n-=1
        
    def TopNode(self)->GNodo:
        return self.__top
    def top(self)->Ind:
        return self.__top.data
    def empty(self)->bool:
        return self.__n == 0
    def size(self)->int:
        return self.__n
    def Resize(self,numSize):
        q:GNodo = self.__top
        n = 0
        while q.Rnode is not None:
            if n == numSize:
                q.Rnode=None
                break
            q = q.Rnode
            n+=1
        self.__n = numSize
    def __getitem__(self, item):
        q:GNodo = self.__top
        n = 0
        while q is not None:
            if n==item:
                return q
            q=q.Rnode
            n+=1
@njit()
def CreateGenPriorityQueue(maxNum):
    return GPQueue(maxNum)
listPQ_type.define(GPQueue.class_type.instance_type)
listPQ_type = deferred_type()