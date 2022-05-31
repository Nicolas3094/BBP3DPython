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
        if self.Cond1(vec,top) or self.Cond2(vec,top) or self.Cond3(vec,top):
            tmp.next = self.__top
            self.__top=tmp
            self.__n+=1
            return
        q = self.__top
        qprev = q
        while q.next is not None:
            qprev = q
            q = q.next
            if self.Cond1(vec,q.data) or self.Cond2(vec,q.data) or self.Cond3(vec,q.data):
                q=qprev
                break
            if (q.data[0] == tmp.data[0] and 
                q.data[1] == tmp.data[1] and 
                q.data[2] == tmp.data[2]):
                return
        tmp.next=q.next 
        q.next = tmp
        self.__n+=1
    def eliminateDuplicates(self):
        q = self.__top
        qprev = q
        while q.next is not None:
            qprev = q
            q = q.next
            if qprev.data[0] == q.data[0] and qprev.data[2] == q.data[2]:
                if q.next is not None:
                    qprev.next = q.next
                    q=qprev
                    self.__n -=1
                else:
                    q=None
                    qprev.next = None
                    self.__n -=1
                    break
    def pop(self)->None:
        if not self.empty():
            tmp = self.__top
            if self.size()==1:
                self.__n=0
                return
            self.__top = tmp.next
            self.__n-=1
    def printQ(self):
        if self.__n == 0:
            return
        q=self.__top
        while q.next is not None:
            print(q.data)
            q=q.next
        print(q.data)
    def getPt(self, i:int):
        q = self.__top
        k = 0
        while q.next is not None and k != i:
            q=q.next
            k+=1
        return q.data
    def delPt(self, i:int):
        q = self.__top
        qprev = q
        k = 0
        if i == 0:
            if q.next is None:
                self.__top=make_linked_node(NumbaList([-1,-1,-1]))
                self.__n = 0
            else:
                self.__top=q.next
                self.__n -= 1
            return
        while q.next is not None and k != i:
            qprev=q
            q=q.next
            k+=1
        if q.next is None:
            qprev.next = None
        else:
            qprev.next = q.next
        self.__n -= 1
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
