import numpy as np
import random
def OX(P1:list[int],P2:list[int],i:int,j:int)->list[int]:
    n = len(P1)
    h1 = [None]*n
    h1[i:j]=P1[i:j]
    for k in np.arange(j,len(P1)):
        for l in np.arange(0,len(P2)):
            if(P2[l] not in h1):
                h1[k] = P2[l]
                break
    for k in np.arange(0,i):
        for l in np.arange(0,len(P2)):
            if(P2[l] not in h1):
                h1[k] = P2[l]
                break 
    return h1
def Cruza(genome1:list[int],genome2:list[int])->tuple[list[int],list[int]]:
    n = len(genome1)
    i= random.randrange(3,int(n/2))
    j= random.randrange(i+1,n)
    h1 = OX(genome1,genome2,i,j)
    h2 = OX(genome2,genome1,i,j)
    return (h1,h2)

def Mutation(gen:list[int], PM:float)->None:
        r = random.random()
        if r >= PM:
            return
        n = len(gen)
        i= random.randrange(0,int(n/2))
        j= random.randrange(i+1,n)
        aux_code = gen[0:i]+gen[i:j][::-1]+gen[j:n]
        gen = list()
        gen = aux_code