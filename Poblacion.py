class Individuo:
    def __init__(self, code:list=None ,BoxSeq:dict=None):
        self.fi=0
        self.code = code
        self.params =None
        if(BoxSeq is not None and code is None):
            self.code= self.__cod__(BoxSeq)
    def __cod__(self,BoxSeq:dict):
        codif = list(BoxSeq.keys())
        return codif
    def AddParameter(self,param):
        self.params = param

class Poblacion(list):
    def __init__(self,num_pop:int):
        self.n = num_pop
        self.prom = 0
        self = list()

    def Add(self,individuo:Individuo):
        self.append(individuo)

    def CreateInd(self, code=None, BoxSeq=None, param=None):
        newInd = Individuo(code=code,BoxSeq=BoxSeq)
        if param is not None:
            newInd.AddParameter(param)
        self.Add(individuo = newInd)
        
    def Convergence(self)->float:
        f_best = self[0].fi
        f_worst = self[-1].fi
        return (f_best - f_worst)/(f_best*f_best)


            