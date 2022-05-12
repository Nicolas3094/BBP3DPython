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
node_type.define(Nodo.class_type.instance_type)




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
        top:List[int]=self.__top.data
        if self.Cond1(vec,top):
            tmp.next = self.__top
            self.__top=tmp
            self.__n+=1
            return
        elif self.Cond2(vec,top):
            tmp.next = self.__top
            self.__top=tmp
            self.__n+=1
            return
        elif self.Cond3(vec,top):
            tmp.next = self.__top
            self.__top=tmp
            self.__n+=1
            return
        q = self.__top
        qprev = q
        while q.next is not None:
            qprev = q
            q = q.next
            if self.Cond1(vec,q.data):
                q=qprev
                break
            elif self.Cond2(vec,q.data):
                q=qprev
                break
            elif self.Cond3(vec,q.data):
                q=qprev
                break
        tmp.next=q.next 
        q.next = tmp
        self.__n+=1
    def pop(self)->None:
        if not self.empty():
            tmp = self.__top
            self.__top = tmp.next
            self.__n-=1

    def top(self)->List[int]:
        return self.__top.data
    def empty(self)->bool:
        return self.__n == 0
    def size(self)->int:
        return self.__n
    def Cond1(self,vec1: List[int],vec2: List[int])->bool:
        return vec1[self.__order[0]] < vec2[self.__order[0]]
    def Cond2(self,vec1: List[int],vec2: List[int])->bool:
        return vec1[self.__order[0]] == vec2[self.__order[0]] and vec1[self.__order[1]] < vec2[self.__order[1]]
    def Cond3(self,vec1: List[int],vec2: List[int])->bool:
        return vec1[self.__order[0]] == vec2[self.__order[0]] and vec1[self.__order[1]] == vec2[self.__order[1]] and  vec1[self.__order[2]] < vec2[self.__order[2]]

@njit
def CreatePriorityQueue(order):
    return PQVector(order)
listP_type.define(PQVector.class_type.instance_type)
listPQ_type = deferred_type()
specPQ = OrderedDict()
specPQ['__n'] = types.int64
specPQ['__top'] = node_type
