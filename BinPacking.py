from Contenedor import Bin,Item
import numpy as np

class BBP:
    def __init__(self,Bins:list = list, Items:list=list,method = None):
        self._bins = Bins
        self._singleP = False
        self._method = method
        if len(Bins) == 1 :
            self._singleP = True
    def AddBin(self, Bin:Bin):
        self._bins.append(Bin)
        if len(self._bins) != 1 :
            self._singleP = False
    def FillContainer(self,index=0,BoxesToPack:list=None, IdBoxes:list=None):
        if self._singleP:
            self._method(
                bin= self._bins[0],
                itemsToPack = BoxesToPack,
                ITEMSDATA = IdBoxes)
            return
        for bin in self._bins:
            self._method(bin,BoxesToPack,IdBoxes)
    def GetBin(self,index=0)->Bin:
        return self._bins[index]
    def GetBins(self)->list:
        return self._bins
    def DefinePackingMethod(self, Function):
        self._method = Function
    def solve(self,index=0,BoxesToPack:list=None,IdBoxes:list=None):
        return self.FillContainer(index = index, BoxesToPack =BoxesToPack,IdBoxes= IdBoxes)  

