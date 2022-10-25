
from numba.typed import List as NumbaList
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

    def CreateDataSet(self,DataSet:str,p:int):
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
        itemsValues = [ list([int(itemV.dimensions[0]),int(itemV.dimensions[1]),int(itemV.dimensions[2])]) for itemV in items]
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
def SimplifiedBoxes(boxes:list[list[int]])->dict:
    visited = []
    nw = {}
    for box in boxes:
        bx = str(box)
        if box not in visited:
            visited.append(box)
            nw[bx]=1
        else:
            nw[bx] +=1
    return nw
def StrToList(boxes:str):
    
    boxes = boxes.replace(" ","")
    boxes = boxes.replace(",",",1,")
    boxes = boxes.replace("[","1,")
    boxes = boxes.replace("]","")
    return list(map(lambda x: int(x),boxes.split(",")))
def createInstance(nm:str,num:int=100):
    nm2= ""
    directory = "Instance/"
    if nm=="P2A2":
        nm2='Alg2-P2'
    elif nm == "P3A2":
        nm2='Alg2-P3'
    elif nm == "P4A2":
        nm2='Alg2-P4'
    elif nm == "P5A2":
        nm2='Alg2-P5'
    else:
        raise("Error")
    generator = BPP3DInstanceGenerator()
    pseed = lambda p : (2502505 + 100*(p - 1))
    n = len(generator.CreateDataSet(nm2,0)[1])
    a = open(directory+nm+ ".csv", 'w')
    savetxt(a,[ [ 100, n]],delimiter=" ",fmt="%d")
    for seed in np.arange(1,num+1):
        data = generator.CreateDataSet(nm2,seed)
        section = [[seed,pseed(seed)]]
        boxes:list[list[int]] = data[1]
        nw = SimplifiedBoxes(boxes)
        kys = list(nw.keys())
        section.append(list(data[0][0]))
        section.append([len(kys)])
        for i in np.arange(len(kys)):
            st = StrToList(kys[i])
            st.insert(0,nw[str([st[1],st[3],st[5]])])
            st.insert(0,i+1)
            section.append(st)
        for i in np.arange(len(section)):
            savetxt(a,[np.array(section[i],dtype=np.int64)],delimiter=" ",fmt="%d")
    a.close()

def createPA():
    createInstance("P2A2")
    createInstance("P3A2")
    createInstance("P4A2")
    createInstance("P5A2")


def CreateInstance(problem:pd.Series)->list[list]:
    if problem[0][0] == " ":
        for i in np.arange(len(problem)):
            problem[i]=problem[i][1:]
    if " " in problem[0]:
        n = int(problem[0].split(" ")[1])
    else:
        n= -1
    bin = np.array(problem[2].split(" "),dtype=np.int64)
    totalboxes=[]
    newcontainer=False
    boxes:list[int] = []
    for i in np.arange(len(problem)):
        res =np.array(problem[i].split(" "),dtype=np.int64)
        k =len(res)
        if k == 8 and not newcontainer:
            if len(boxes) != 0: boxes.clear()
            newcontainer=True
        elif k!=8 and newcontainer:
            newcontainer=False
            if len(boxes) != n and n != -1:
                raise("AssertionError")
            totalboxes.append(NumbaList(boxes))
            boxes=list()
        if newcontainer:
            for k in np.arange(res[1]):
                boxes.append(NumbaList([ # (orientacion valita Vertical, Medida de caja)
                    NumbaList([res[3],res[2]]), 
                     NumbaList([res[5],res[4]]),
                     NumbaList([res[7],res[6]])
                    ]))
        if i==len(problem)-1:
            totalboxes.append(boxes)
    return [NumbaList(bin),np.array(totalboxes,dtype=np.int64)]

def GetInstance(nm:str):
    if nm=="P2A2":
        return CreateInstance(pd.read_csv("Instance/P2A2.csv",header=None)[0])
    elif nm=="P3A2":
        return CreateInstance(pd.read_csv("Instance/P3A2.csv",header=None)[0])
    elif nm=="P4A2":
        return CreateInstance(pd.read_csv("Instance/P4A2.csv",header=None)[0])
    elif nm=="P5A2":
        return CreateInstance(pd.read_csv("Instance/P5A2.csv",header=None)[0])
    elif nm=="BR1":
        return  CreateInstance(pd.read_csv("Instance/BR1.csv",header=None)[0])
    elif nm=="BR2":
        return  CreateInstance(pd.read_csv("Instance/BR2.csv",header=None)[0])
    elif nm=="BR3":
        return  CreateInstance(pd.read_csv("Instance/BR3.csv",header=None)[0])
    elif nm=="BR4":
        return  CreateInstance(pd.read_csv("Instance/BR4.csv",header=None)[0])
    elif nm=="BR5":
        return  CreateInstance(pd.read_csv("Instance/BR5.csv",header=None)[0])
    elif nm=="BR6":
        return  CreateInstance(pd.read_csv("Instance/BR6.csv",header=None)[0])
    elif nm=="BR7":
        return  CreateInstance(pd.read_csv("Instance/BR7.csv",header=None)[0])
