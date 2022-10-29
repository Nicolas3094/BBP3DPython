from numba import types,njit,deferred_type, optional
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from typing import List
from collections import OrderedDict
node_type = deferred_type()
specNode = OrderedDict()
specNode['data'] = types.ListType(types.int64)
specNode['next'] = optional(node_type)
@jitclass(specNode)
class Nodo:
    def __init__(self,data:List[int],nodo):
        self.data = data
        self.next = nodo
    def prepend(self, data):
        return Nodo(data, self)
@njit
def make_linked_node(data):
    return Nodo(data, None)
node_type.define(Nodo.class_type.instance_type)  # type: ignore

listP_type = deferred_type()
specP = OrderedDict()
specP['__order'] = types.ListType(types.int64)
specP['__n'] = types.int64
specP['__top'] = node_type
@jitclass(specP)
class PQVector:
    def __init__(self, order: List[int]):
        self.__order= order
        self.__n = 0
        self.__top=make_linked_node(NumbaList([-1,-1,-1])) 
    def push(self, vec: List[int])->None:
        if self.empty():
            self.__top=make_linked_node(NumbaList(vec))           
            self.__n+=1
            return 
        tmp = make_linked_node(vec)
        q = self.__top
        if self.__Condition__(vec,self.__top.data):
            tmp.next = self.__top
            self.__top=tmp
            self.__n+=1
            return
        if q.data == tmp.data:
            return 
        qprev = q
        while q.next is not None:
            qprev = q
            q = q.next
            if self.__Condition__(vec,q.data):
                q=qprev
                break
            if q.data == tmp.data:
                return
        tmp.next=q.next 
        q.next = tmp
        self.__n+=1
    def updateList(self)->None:
        q = self.__top
        while q.next is not None:
            qprev = q
            q = q.next
            if  (qprev.data[0] == q.data[0] and  qprev.data[2] == q.data[2]) or  (qprev.data[0] < q.data[0] and  qprev.data[1] == q.data[1]  and qprev.data[2] == q.data[2]):
                qprev.next = q.next
                self.__n -=1
    def pop(self)->None:
        if not self.empty():
            tmp = self.__top
            if self.size()==1:
                self.__n=0
                return
            self.__top = tmp.next
            self.__n-=1
    def ToList(self)->List[List[int]]:
        ls = []
        q = self.__top
        while q.next is not None:
            ls.append(NumbaList(q.data))
            q = q.next
        ls.append(NumbaList(q.data))
        return NumbaList(ls)
    def getPt(self, i:int)->List[int]:
        q = self.__top
        k = 0
        while q.next is not None and k != i:
            q=q.next
            k+=1
        return q.data
    def changePt(self,i:int,val:List[int]):
        q = self.__top
        k = 0
        while q.next is not None and k != i:
            q=q.next
            k+=1
        q.data = val
    def delPt(self, i:int)->None:
        q = self.__top
        qprev = q
        k = 0
        if i == 0:
            return self.pop()
        while q.next is not None and k != i:
            qprev=q
            q=q.next
            k+=1
        qprev.next = q.next
        self.__n -= 1
    def top(self)->List[int]:
        return self.__top.data
    def empty(self)->bool:
        return self.__n == 0
    def clear(self):
        while not self.empty():
            self.pop()
    def size(self)->int:
        return self.__n
    def __Condition__(self,vec1:List[int],vec2:List[int])->bool:
        return self.__Cond3__(vec1,vec2) or self.__Cond1__(vec1,vec2) or self.__Cond2__(vec1,vec2)
    def __Cond1__(self,vec1: List[int],vec2: List[int])->bool:
        return vec1[self.__order[0]] < vec2[self.__order[0]]
    def __Cond2__(self,vec1: List[int],vec2: List[int])->bool:
        return vec1[self.__order[0]] == vec2[self.__order[0]] and vec1[self.__order[1]] < vec2[self.__order[1]]
    def __Cond3__(self,vec1: List[int],vec2: List[int])->bool:
        return vec1[self.__order[0]] == vec2[self.__order[0]] and vec1[self.__order[1]] == vec2[self.__order[1]] and  vec1[self.__order[2]] < vec2[self.__order[2]]

@njit
def CreatePriorityQueue(order):
    return PQVector(order)
listP_type.define(PQVector.class_type.instance_type)  # type: ignore
