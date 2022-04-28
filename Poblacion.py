from typing import List

Genome = List[int]
Population = List[Genome]

class Individuo:
    def __init__(self, code:Genome):
        self.fi=None
        self.genome:Genome = code
        self.params = None
    def AddParameter(self,param):
        self.params = param
    def __str__(self) -> str:
        return ",".join(map(str,self.genome))
    def __eq__(self, other: list) -> bool:
        return self.genome == other
    def __getitem__(self, item)->int:
        return self.genome[item]

class Poblation:
    def __init__(self, num:int=None, genomes:Population = None):
        self.poblation:Population = []
        self.n = num
        if genomes is not None and num is None:
            self.poblation:Population = genomes
            self.n = len(genomes)

    def Add(self,individuo:Individuo):
        self.poblation.append(individuo)

    def CreateInd(self, code:Genome=None):
        newInd = Individuo(code=code)
        self.Add(newInd)
    def DeleteInd(self, index:int):
        del self.poblation[index]
    def __getitem__(self, item)->Individuo:
        return self.poblation[item]

    def __str__(self) -> str:
        return "\n".join(map(str,self.poblation))


            