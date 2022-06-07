
from ast import Lambda
import random
import pandas as pd
from numpy import savetxt
import numpy as np
class Item:
    def __init__(self,classType=0,dimensions=None, minMaxVertex = None):
        self.id = classType
        self.dimensions = np.array(dimensions)
        if minMaxVertex is not None:
            self.MinV = np.array(minMaxVertex[0])
            self.MaxV = np.array(minMaxVertex[1])
            self.dimensions = np.abs(self.MaxV-self.MinV)
        else:
             self.MinV =None
             self.MaxV=None
        self.Vol = np.prod(self.dimensions)
    def AddPoint(self, point:list):
        self.MinV = np.array(point)
        self.MaxV = self.dimensions + self.MinV

class BPP3DInstanceGenerator():
    def __init__(self):
       pass

    def CreateDataSet(self,DataSet,p):
        np.random.seed(2502505 + 100*(p - 1))
        AlgBinsProblem = np.array(
                [[20,20,20],
                [50,50,50],
                [100,100,100],
                [200,200,200],
                [300,300,300]],
            dtype=np.int64)
        createDataSet1 = lambda alg, num : [
            np.array([AlgBinsProblem[alg]],dtype=np.int64),
            self.Alg1(num,AlgBinsProblem[alg])]   
        createDataSet2 = lambda alg, num : [
            np.array([AlgBinsProblem[alg]],dtype=np.int64),
            self.Alg2(num,AlgBinsProblem[alg])]  
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
            return createDataSet1(0,15)
        elif DataSet=='Alg2-P1':
            return createDataSet2(0,16)
            
        elif DataSet=='Alg1-P2':
            return createDataSet1(1,29)
        elif DataSet=='Alg2-P2':
            return createDataSet2(1,25)
        
        elif DataSet=='Alg1-P3':
            return createDataSet1(2,50)
        elif DataSet=='Alg2-P3':
            return createDataSet2(2,52)
      
        elif DataSet=='Alg1-P4':
            return createDataSet1(3,106)
        elif DataSet=='Alg2-P4':
            return createDataSet2(3,100)
               
        elif DataSet=='Alg1-P5':
            return [[AlgBinsProblem[4]],self.Alg1(155,AlgBinsProblem[4])]
        elif DataSet=='Alg2-P5':
            return createDataSet2(4,151)
        
       
        else:
            raise Exception("DataSets: S1,..., S8, Alg1-P1,..., Alg3-P1,...Alg3-P5")
        
    def Alg1(self,num_boxes,bin)->list[list[int]]:
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
        itemsValues = np.array([ itemV.dimensions for itemV in items],dtype=np.int64)
        return itemsValues
    
    def Alg2(self,num_boxes,bin)->list[list[int]]:
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
        itemsValues = np.array([ itemV.dimensions for itemV in items],dtype=np.int64)
        return itemsValues

    def Alg3(self,num_boxes,bin):
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
        #np.random.seed(2502505 + 100*(p - 1))

        if type==1:
            dj,wj,hj = np.random.uniform((2/3)*D,D), np.random.uniform(1,(1/2)*W), np.random.uniform((2/3)*H,H)
        elif type==2:
            dj,wj,hj = np.random.uniform((2/3)*D,D), np.random.uniform((2/3)*W,W),np.random.uniform(1,(1/2)*H)
        elif type==3:
            dj,wj,hj = np.random.uniform(1,(1/2)*D), np.random.uniform((2/3)*W,W), np.random.uniform((2/3)*H,H)
        elif type==4:
            dj,wj,hj = np.random.uniform((1/2)*D,D), np.random.uniform((1/2)*W,W), np.random.uniform((1/2)*H,H)
        elif type==5:
            dj,wj,hj = np.random.uniform(1,(1/2)*D), np.random.uniform(1,(1/2)*W), np.random.uniform(1,(1/2)*H) 
        elif type == 6:
            dj,wj,hj = np.random.uniform(1,10), np.random.uniform(1,10),  np.random.uniform(1,10)
        elif type == 7:
            dj,wj,hj = np.random.uniform(1,35) , np.random.uniform(1,35), np.random.uniform(1,35)           
        elif type == 8:
            dj,wj,hj = np.random.uniform(1,100),np.random.uniform(1,100),np.random.uniform(1,100)
        else:
            return self.CreateItemsType(type=1,BIN=[D,W,H])
        return np.asarray([dj,wj,hj],dtype=np.int64)

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

createDATA = lambda x,y : savetxt(
    fname = "Instance/"+x + ".csv",
    X= y,
    delimiter=", ",
    fmt='%g'
)

#Martello (200) I-IV-V, Berkey and Wang (1987) VI-VII-VIII
classI = lambda ClassNum,BoxNum,InstanceNum :  np.asanyarray(pd.read_csv("Instance/Class"+ClassNum+"/"+str(BoxNum)+"/"+str(InstanceNum)+".csv",delimiter=",",header=None),dtype=np.int64)

BIN_CLSI = np.array([100,100,100],dtype=np.int64)
CLASS_I50 =lambda InstaceID: classI(ClassNum="I",BoxNum = 50,InstanceNum=InstaceID)
CLASS_I100 =lambda InstaceID: classI(ClassNum="I",BoxNum = 100,InstanceNum=InstaceID)
CLASS_I150 =lambda InstaceID: classI(ClassNum="I",BoxNum = 150,InstanceNum=InstaceID)
CLASS_I200 =lambda InstaceID: classI(ClassNum="I",BoxNum = 200,InstanceNum=InstaceID)

BIN_CLSIV = np.array([100,100,100],dtype=np.int64)
CLASS_IV50 =lambda InstaceID: classI(ClassNum="IV",BoxNum = 50,InstanceNum=InstaceID)
CLASS_I1V00 =lambda InstaceID: classI(ClassNum="IV",BoxNum = 100,InstanceNum=InstaceID)
CLASS_IV50 =lambda InstaceID: classI(ClassNum="IV",BoxNum = 150,InstanceNum=InstaceID)
CLASS_IV200 =lambda InstaceID: classI(ClassNum="IV",BoxNum = 200,InstanceNum=InstaceID)

BIN_CLSV = np.array([100,100,100],dtype=np.int64)
CLASS_V50 =lambda InstaceID: classI(ClassNum="V",BoxNum = 50,InstanceNum=InstaceID)
CLASS_V100 =lambda InstaceID: classI(ClassNum="V",BoxNum = 100,InstanceNum=InstaceID)
CLASS_V50 =lambda InstaceID: classI(ClassNum="V",boxNum = 150,InstanceNum=InstaceID)
CLASS_V200 =lambda InstaceID: classI(ClassNum="V",BoxNum = 200,InstanceNum=InstaceID)


BIN_CLSVI = np.array([10,10,10],dtype=np.int64)
CLASS_VI50 =lambda InstaceID: classI(ClassNum="VI",BoxNum = 50,InstanceNum=InstaceID)
CLASS_VI100 =lambda InstaceID: classI(ClassNum="VI",BoxNum = 100,InstanceNum=InstaceID)
CLASS_VI50 =lambda InstaceID: classI(ClassNum="VI",BoxNum = 150,InstanceNum=InstaceID)
CLASS_VI200 =lambda InstaceID: classI(ClassNum="VI",BoxNum = 200,InstanceNum=InstaceID)


BIN_CLSVII = np.array([40,40,40],dtype=np.int64)
CLASS_VII50 =lambda InstaceID: classI(ClassNum="VII",BoxNum = 50,InstanceNum=InstaceID)
CLASS_VII100 =lambda InstaceID: classI(ClassNum="VII",BoxNum = 100,InstanceNum=InstaceID)
CLASS_VII50 =lambda InstaceID: classI(ClassNum="VII",BoxNum = 150,InstanceNum=InstaceID)
CLASS_VII200 =lambda InstaceID: classI(ClassNum="VII",BoxNum = 200,InstanceNum=InstaceID)

BIN_CLSVIII = np.array([100,100,100],dtype=np.int64)
CLASS_VIII50 =lambda InstaceID: classI(ClassNum="VIII",BoxNum = 50,InstanceNum=InstaceID)
CLASS_VIII100 =lambda InstaceID: classI(ClassNum="VIII",BoxNum = 100,InstanceNum=InstaceID)
CLASS_VIII50 =lambda InstaceID: classI(ClassNum="VIII",BoxNum = 150,InstanceNum=InstaceID)
CLASS_VIII200 =lambda InstaceID: classI(ClassNum="VIII",BoxNum = 200,InstanceNum=InstaceID)


#Instancias de Karabulut
P1A1 =lambda instanceNum: np.asanyarray(pd.read_csv("Instance/P1/A1/"+str(instanceNum)+".csv",delimiter=",",header=None),dtype=np.int64)
P1A2 = lambda instanceNum: np.asanyarray(pd.read_csv("Instance/P1/A2"+str(instanceNum)+".csv",delimiter=",",header=None),dtype=np.int64)

P2A1 = lambda instanceNum: np.asanyarray(pd.read_csv("Instance/P2/A1/"+str(instanceNum)+".csv",delimiter=",",header=None),dtype=np.int64)
P2A2 = lambda instanceNum: np.asanyarray(pd.read_csv("Instance/P2/A2/"+str(instanceNum)+".csv",delimiter=",",header=None),dtype=np.int64)

P3A1 = lambda instanceNum: np.asanyarray(pd.read_csv("Instance/P3/A1/"+str(instanceNum)+".csv",delimiter=",",header=None),dtype=np.int64)
P3A2 = lambda instanceNum: np.asanyarray(pd.read_csv("Instance/P3/A2/"+str(instanceNum)+".csv",delimiter=",",header=None),dtype=np.int64)

P4A1 = lambda instanceNum: np.asanyarray(pd.read_csv("Instance/P4/A1/"+str(instanceNum)+".csv",delimiter=",",header=None),dtype=np.int64)
P4A2 = lambda instanceNum: np.asanyarray(pd.read_csv("Instance/P4/A2/"+str(instanceNum)+".csv",delimiter=",",header=None),dtype=np.int64)

P5A1 =lambda instanceNum: np.asanyarray(pd.read_csv("Instance/P5/A1/"+str(instanceNum)+".csv",delimiter=",",header=None),dtype=np.int64)
P5A2 =lambda instanceNum: np.asanyarray(pd.read_csv("Instance/P5/A2/"+str(instanceNum)+".csv",delimiter=",",header=None),dtype=np.int64)

BINP1 = np.array(np.array(pd.read_csv("Instance/P1/BIN.csv",delimiter=",",header=None),dtype=np.int64)[0],dtype=np.int64)
BINP2= np.array(np.array(pd.read_csv("Instance/P2/BIN.csv",delimiter=",",header=None),dtype=np.int64)[0],dtype=np.int64)
BINP3 = np.array(np.array(pd.read_csv("Instance/P3/BIN.csv",delimiter=",",header=None),dtype=np.int64)[0],dtype=np.int64)
BINP4 = np.array(np.array(pd.read_csv("Instance/P4/BIN.csv",delimiter=",",header=None),dtype=np.int64)[0],dtype=np.int64)
BINP5 = np.array(np.array(pd.read_csv("Instance/P5/BIN.csv",delimiter=",",header=None),dtype=np.int64)[0],dtype=np.int64)

P1A1D = np.array([np.array(pd.read_csv("Instance/P1/A1/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64)
P1A2D = np.array([np.array(pd.read_csv("Instance/P1/A2/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64)

P2A1D = np.array([np.array(pd.read_csv("Instance/P2/A1/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64)
P2A2D = np.array([np.array(pd.read_csv("Instance/P2/A2/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64)

P3A1D = np.array([np.array(pd.read_csv("Instance/P3/A1/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64)
P3A2D = np.array([np.array(pd.read_csv("Instance/P3/A2/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64)

P4A1D =  np.array([np.array(pd.read_csv("Instance/P4/A1/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64)
P4A2D =  np.array([np.array(pd.read_csv("Instance/P4/A2/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64)

P5A1D = np.array([np.array(pd.read_csv("Instance/P5/A1/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64)
P5A2D = np.array([np.array(pd.read_csv("Instance/P5/A2/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64)




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


def CreateData(algorithm:int,problem:int):
    if problem ==1:
        return (BINP1,np.array([np.array(pd.read_csv("Instance/P"+str(problem)+"/A"+str(algorithm)+"/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64))
    elif problem == 2:
        return (BINP2,np.array([np.array(pd.read_csv("Instance/P"+str(problem)+"/A"+str(algorithm)+"/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64))
    elif problem == 3:
        return (BINP3,np.array([np.array(pd.read_csv("Instance/P"+str(problem)+"/A"+str(algorithm)+"/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64))
    elif problem == 4:
        return (BINP4,np.array([np.array(pd.read_csv("Instance/P"+str(problem)+"/A"+str(algorithm)+"/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64))
    else:
        return (BINP5,np.array([np.array(pd.read_csv("Instance/P"+str(problem)+"/A"+str(algorithm)+"/"+str(i+1)+".csv",delimiter=",",header=None),dtype=np.int64) for i in np.arange(20)],dtype=np.int64))
