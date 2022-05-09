class Nodo:
    def __self__(self,valor,nodo=None):
        self.valor = valor
        self.nodo:Nodo = nodo

class PQVector:
    def __init__(self, order: list[int]):
        self._order= order
        self._n = 0
        self._top=None
    def push(self, vec: list[int]):
        tmp = Nodo()
        tmp.valor=vec
        if self.empty():
            tmp.nodo = self._top
            self._top=tmp
            self._n+=1
            return 
        top=self._top.valor
        if self.Cond1(vec,top):
            tmp.nodo = self._top
            self._top=tmp
            self._n+=1
            return
        elif self.Cond2(vec,top):
            tmp.nodo = self._top
            self._top=tmp
            self._n+=1
            return
        elif self.Cond3(vec,top):
            tmp.nodo = self._top
            self._top=tmp
            self._n+=1
            return
        q = self._top
        qprev = q
        while q.nodo is not None:
            qprev = q
            q = q.nodo
            if self.Cond1(vec,q.valor):
                q=qprev
                break
            elif self.Cond2(vec,q.valor):
                q=qprev
                break
            elif self.Cond3(vec,q.valor):
                q=qprev
                break
        tmp.nodo=q.nodo 
        q.nodo = tmp
        self._n+=1
    def pop(self)->list[int]:
        if self.empty():
            return
        value = self._top.valor
        tmp = Nodo()
        tmp = self._top
        self._top = tmp.nodo
        self._n-=1
        return value
    def size(self):
        return self._n
    def Top(self)->list[int]:
        return self._top.valor
    def empty(self):
        return self._n == 0
    def ConvertToList(self)->list:
        lista = []
        aux = self._top
        while aux.nodo is not None:
            lista.append(aux.valor)
            aux = aux.nodo
        return lista
    def Cond1(self,vec1,vec2):
        return vec1[self._order[0]] < vec2[self._order[0]]
    def Cond2(self,vec1,vec2):
        return vec1[self._order[0]] == vec2[self._order[0]] and vec1[self._order[1]] < vec2[self._order[1]]
    def Cond3(self,vec1,vec2):
        return vec1[self._order[0]] == vec2[self._order[0]] and vec1[self._order[1]] == vec2[self._order[1]] and  vec1[self._order[2]] < vec2[self._order[2]]
    def __str__(self) -> str:
        string = ""
        aux = self._top
        while aux.nodo is not None:
            item = aux.valor
            output = [str(x) for x in item]
            stri = ",".join(output)+" -> "
            string += stri
            aux = aux.nodo
        output = [str(x) for x in aux.valor]
        stri = ",".join(output)+" -> "
        string += stri
        string += "None"
        return string