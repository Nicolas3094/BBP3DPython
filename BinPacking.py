from Contenedor import Bin,Item
import numpy as np
import random
import pandas as pd
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


class BPP3DInstanceGenerator():
    def __init__(self, num_items = None, bin=None):
        self._numItems = num_items
        self._Bin = bin

    def CreateDataSet(self,DataSet):
        AlgBinsProblem = [[20,20,20],[50,50,50],[100,100,100],[200,200,200],[300,300,300]]
        if DataSet=='S1':
            return self.SDataSet(1)
        elif DataSet=='S2':      
            return self.SDataSet(2)
        elif DataSet=='S3':      
            return self.SDataSet(3)
        elif DataSet=='S4':      
            return self.SDataSet(4)
        elif DataSet=='S5':      
            return self.SDataSet(5)
        elif DataSet=='S6':      
            return self.SDataSet(6)
        elif DataSet=='S7':      
            return self.SDataSet(7)
        elif DataSet=='S8':      
            return self.SDataSet(8)

        elif DataSet=='Alg1-P1':
            return (AlgBinsProblem[0],self.Alg1(15,AlgBinsProblem[0]))
        elif DataSet=='Alg2-P1':
            return (AlgBinsProblem[0],self.Alg2(16,AlgBinsProblem[0]))
        elif DataSet=='Alg1-P2':
            return (AlgBinsProblem[1],self.Alg1(29,AlgBinsProblem[1]))
        elif DataSet=='Alg2-P2':
            return (AlgBinsProblem[1],self.Alg2(25,AlgBinsProblem[1]))
        
        elif DataSet=='Alg1-P3':
            return (AlgBinsProblem[2],self.Alg1(50,AlgBinsProblem[2]))
        elif DataSet=='Alg2-P3':
            return (AlgBinsProblem[2],self.Alg2(52,AlgBinsProblem[2]))
      

        elif DataSet=='Alg1-P4':
            return (AlgBinsProblem[3],self.Alg1(106,AlgBinsProblem[3]))
        elif DataSet=='Alg2-P4':
            return (AlgBinsProblem[3],self.Alg2(100,AlgBinsProblem[3]))
       
        
        elif DataSet=='Alg1-P5':
            return (AlgBinsProblem[4],self.Alg1(155,AlgBinsProblem[4]))
        elif DataSet=='Alg2-P5':
            return (AlgBinsProblem[4],self.Alg2(151,AlgBinsProblem[4]))
       
        else:
            raise Exception("DataSets: S1,..., S8, Alg1-P1,..., Alg3-P1,...Alg3-P5")
        
    def Alg1(self,num_boxes = None,bin= None)->list:
        if num_boxes == None or bin == None:
            num_boxes = self._numItems
            bin = self._Bin
        i= int(np.ceil((num_boxes-1)/7))
        items=[Item(minMaxVertex=([0,0,0],bin))]
        for _ in range(i):
            j = np.random.randint(len(items))
            item:Item = items[j]
            while item.dimensions[0]<=1 or item.dimensions[1]<=1 or item.dimensions[2]<=1:
                j = np.random.randint(len(items))
                item = items[j]
            del items[j]
            randpoint = [0 for i in range(3)]
            for k in range(3):
                randpoint[k] = np.random.randint(1,item.dimensions[k])
                while(randpoint[k]==0):
                    randpoint[k] = np.random.randint(1,item.dimensions[k])
            newCubes = self.__Create8Boxes__(randpoint,item.dimensions)
            items = items + newCubes
        itemsValues = [ itemV.dimensions for itemV in items]
        return itemsValues
    
    def Alg2(self,num_boxes = None,bin= None):
        if num_boxes == None or bin == None:
            num_boxes = self._numItems
            bin = self._Bin
        i= int(np.ceil((num_boxes-1)/3))
        items=[Item(minMaxVertex=([0,0,0],bin))]
        for _ in range(i):
            j = np.random.randint(len(items))
            item = items[j]
            while item.dimensions[0]<=1 or item.dimensions[1]<=1 or item.dimensions[2]<=1:
                j = np.random.randint(len(items))
                item = items[j]
            del items[j]
            randomAxis = np.random.randint(2)
            randomAxisPoint =  np.random.randint(1,item.dimensions[randomAxis])
            randpoint = np.zeros(3)
            randpoint[randomAxis] = randomAxisPoint
            randpoint[2] = np.random.randint(1,item.dimensions[2])
            newCubes = self.__Create4Boxes__(randpoint,item.dimensions)
            items = items + newCubes
        itemsValues = [ itemV.dimensions for itemV in items]
        return itemsValues

    def Alg3(self,num_boxes = None,bin= None):
        if num_boxes == None or bin == None:
            num_boxes = self._numItems
            bin = self._Bin
        i= int(np.ceil((num_boxes-1)/10))
        items=[Item(minMaxVertex=([0,0,0],bin))]
        for _ in range(i):
            j = np.random.randint(len(items))
            item = items[j]
            while item.dimensions[0]<=1 or item.dimensions[1]<=1 or item.dimensions[2]<=1:
                j = np.random.randint(len(items))
                item = items[j]
            del items[j]
            randpoint1 = np.array([np.random.randint(1,item.dimensions[k]) for k in range(3)])
            randpoint2 = np.array([np.random.randint(1,item.dimensions[k]) for k in range(3)])
            while randpoint1[0] == randpoint2[0] and randpoint1[1] == randpoint2[1] and randpoint1[2] == randpoint2[2]:
                randpoint2 = [np.random.randint(1,item.dimensions[k]) for k in range(3)]
            maxit,minit = None,None
            if np.sqrt(randpoint1.dot(randpoint1)) > np.sqrt(randpoint2.dot(randpoint2)):
                maxit = randpoint1
                minit = randpoint2
            else:
                maxit = randpoint2
                minit = randpoint1
            
            smallBox = Item(minMaxVertex=[minit,maxit])
            items.append(smallBox)

            #items.append(Item(minMaxVertex=[maxit,[0,0,item.dimensions[2]]]))
            #items.append(Item(minMaxVertex=[maxit,[0,item.dimensions[1],0]]))
            #items.append(Item(minMaxVertex=[maxit,[item.dimensions[0],0,0]]))

            #items.append(Item(minMaxVertex=[maxit,[0,0,item.dimensions[2]]]))
            #items.append(Item(minMaxVertex=[maxit,[0,item.dimensions[1],0]]))
            #items.append(Item(minMaxVertex=[maxit,[item.dimensions[0],0,0]]))

            print(minit,maxit)

            #break
            #newCubes = Create4Boxes(randpoint,item.dimensions)
            #items = items + newCubes
        itemsValues = [ itemV.dimensions for itemV in items]
        return itemsValues

    def __Create8Boxes__(self,pointCut:list,MaxValue:list):
        items = []
        x0,y0,z0 = pointCut[0],pointCut[1],pointCut[2]
        X,Y,Z = MaxValue[0],MaxValue[1],MaxValue[2]
        
        items.append(Item(minMaxVertex=([0,0,0],
                                        pointCut))) #cube 1
        items.append(Item(minMaxVertex=([0,y0,0],
                                        [x0,Y,z0]))) #cube 2
        items.append(Item(minMaxVertex=([0,0,z0],
                                        [x0,y0,Z]))) #cube 3
        items.append(Item(minMaxVertex=([0,y0,z0],
                                        [x0,Y,Z]))) #cube 4

        items.append(Item(minMaxVertex=([x0,0,0],
                                        [X,y0,z0]))) #cube 5
        items.append(Item(minMaxVertex=([x0,y0,0],
                                        [X,Y,z0]))) #cube 6
        items.append(Item(minMaxVertex=([x0,0,z0],
                                        [X,y0,Z]))) #cube 7
        items.append(Item(minMaxVertex=(pointCut,
                                        MaxValue))) #cube 8
        return items

    def __Create4Boxes__(self,pointCut:list,MaxValue:list):
        items = []
        x0,y0,z0 = pointCut[0],pointCut[1],pointCut[2]
        X,Y,Z = MaxValue[0],MaxValue[1],MaxValue[2]
        if pointCut[0]!=0:
            items.append(Item(minMaxVertex=([0,0,0], #cube 1
                                            [x0,Y,z0]))) 
            items.append(Item(minMaxVertex=([0,0,z0], #cube 2
                                            [x0,Y,Z]))) 
            items.append(Item(minMaxVertex=([x0,0,0], #cube 3
                                            [X,Y,z0]))) 
            items.append(Item(minMaxVertex=([x0,0,z0], #cube 4
                                            MaxValue))) 
        elif pointCut[1]!=0:
            items.append(Item(minMaxVertex=([0,0,0], #cube 1
                                            [X,y0,z0]))) 
            items.append(Item(minMaxVertex=([0,y0,0], #cube 2
                                            [X,Y,z0]))) 
            items.append(Item(minMaxVertex=([0,0,z0], #cube 3
                                            [X,y0,Z]))) 
            items.append(Item(minMaxVertex=([0,y0,z0], #cube 4
                                            MaxValue))) 
        return items

    def CreateItemsType(self, type:int, BIN:list):
        D,W,H =BIN[0],BIN[1],BIN[2]
        wj,dj,hj = (0,0,0)
        if type==1:
            dj,wj,j = random.uniform((2/3)*D,D), random.uniform(1,(1/2)*W), random.uniform((2/3)*H,H)
        elif type==2:
            dj,wj,hj = random.uniform((2/3)*D,D), random.uniform((2/3)*W,W), random.uniform(1,(1/2)*H)
        elif type==3:
            dj,wj,hj = random.uniform(1,(1/2)*D), random.uniform((2/3)*W,W), random.uniform((2/3)*H,H)
        elif type==4:
            dj,wj,hj = random.uniform((1/2)*D,D), random.uniform((1/2)*W,W), random.uniform((1/2)*H,H)
        elif type==5:
            dj,wj,hj = random.uniform(1,(1/2)*D), random.uniform(1,(1/2)*W), random.uniform(1,(1/2)*H) 
        elif type == 6:
            dj,wj,hj = random.uniform(1,10), random.uniform(1,10),  random.uniform(1,10)
        elif type == 7:
            dj,wj,hj = random.uniform(1,35) , random.uniform(1,35), random.uniform(1,35)           
        elif type == 8:
            dj = random.uniform(1,100),random.uniform(1,100),random.uniform(1,100)
        else:
            return self.CreateItemsType(1,W,H,D)
        return [wj,dj,hj]

    def SDataSet(self, num):
        Bin1 = [100,100,100]
        Bin2 = [1000,1000,1000]
        bins = [Bin1,Bin1,Bin1,Bin1,Bin2,Bin2,Bin2,Bin2]
        betas = [0.15,0.15,0.25,0.05,0.05,0.05,0.05,0.05]
        gammas =[0.85,0.50,0.75,0.50,0.85,0.85,0.85,0.85]
        ms = [5,5,5,10,5,10,20,30]
        data = []
        num-=1
        auxBin = bins[num]
        bin = Bin2
        if num<4 and num>0:
            bin = Bin1
        for _ in range(ms[num]):
            dim = [auxBin[j]*random.uniform(betas[num],gammas[num]) for j in range(3)]
            data.append(dim)            
        return (bin,data)
generator = BPP3DInstanceGenerator()
PROBLEM1 = [generator.CreateDataSet('Alg1-P1'),generator.CreateDataSet('Alg2-P1')]
PROBLEM2 = [generator.CreateDataSet('Alg1-P2'),generator.CreateDataSet('Alg2-P2')]
PROBLEM3 = [generator.CreateDataSet('Alg1-P3'),generator.CreateDataSet('Alg2-P3')]
PROBLEM4 = [generator.CreateDataSet('Alg1-P4'),generator.CreateDataSet('Alg2-P4')]
PROBLEM5 = [generator.CreateDataSet('Alg1-P5'),generator.CreateDataSet('Alg2-P5')]
UDLPROBLEM = {
    'UDL1': {
        'dimension' : np.asanyarray(pd.read_csv("BPP_data/data/ULDs/uld1.csv",sep=';', header=None)[[0,1,2]])[0],
        '8boxes':[
                np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/8boxes/box0.csv",sep=';', header=None)[[0,1,2]]),
                np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/8boxes/box1.csv",sep=';', header=None)[[0,1,2]]),
                np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/8boxes/box2.csv",sep=';', header=None)[[0,1,2]]),
                np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/8boxes/box3.csv",sep=';', header=None)[[0,1,2]]),
                np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/8boxes/box4.csv",sep=';', header=None)[[0,1,2]])
            ]
        ,
        '12boxes' : [
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/12boxes/box0.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/12boxes/box1.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/12boxes/box2.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/12boxes/box3.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/12boxes/box4.csv",sep=';', header=None)[[0,1,2]])
                ]
        ,
        '18boxes':[
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/18boxes/box0.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/18boxes/box1.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/18boxes/box2.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/18boxes/box3.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/18boxes/box4.csv",sep=';', header=None)[[0,1,2]])
                    ]
        ,
        '27boxes':[
                   np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/27boxes/box0.csv",sep=';', header=None)[[0,1,2]]),
                   np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/27boxes/box1.csv",sep=';', header=None)[[0,1,2]]),
                   np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/27boxes/box2.csv",sep=';', header=None)[[0,1,2]]),
                   np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/27boxes/box3.csv",sep=';', header=None)[[0,1,2]]),
                   np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/1ULD/27boxes/box4.csv",sep=';', header=None)[[0,1,2]])
                ]    
    },
    'UDL2': {
        'dimension' : np.asanyarray(pd.read_csv("BPP_data/data/ULDs/uld2.csv",sep=';', header=None)[[0,1,2]])[0],
        '16boxes':[
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/16boxes/box0.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/16boxes/box1.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/16boxes/box2.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/16boxes/box3.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/16boxes/box4.csv",sep=';', header=None)[[0,1,2]])
                    ]
            ,
        '20boxes':[
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/20boxes/box0.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/20boxes/box1.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/20boxes/box2.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/20boxes/box3.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/20boxes/box4.csv",sep=';', header=None)[[0,1,2]])
                    ]
            ,
        '24boxes':[
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/24boxes/box0.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/24boxes/box1.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/24boxes/box2.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/24boxes/box3.csv",sep=';', header=None)[[0,1,2]]),
                    np.asanyarray(pd.read_csv("BPP_data/data/Boxes/series1/2ULD/24boxes/box4.csv",sep=';', header=None)[[0,1,2]])
                    ]   
    }
}