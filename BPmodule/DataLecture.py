import json
import os
from BPnumba.BoxN import ItemBin
from BPnumba.BPPH import Bin

def ExpotarJSON(path:str,rot:int,Cont:Bin,Data:list[ItemBin]):
    
    binDim = Cont.dimensions
    n=Cont.getN()
    gen = Cont.getBoxes()
    rgen = Cont.getRot()
    positions = Cont.getPositions()
    data = {
      "box": [],
      "bin": {
        "x":int(binDim[0]),"y":int(binDim[2]),"z":int(binDim[1])
      }
      }
    for i in range(n):
        
        cajaID = gen[i]-1
        
        item = Data[cajaID]
        if rot != 0:
          item.rotate(rgen[i],rot)
        
        itemDim = item.CDim()
        data["box"].append({
        "id": item.T,
        "dimension": {"x":int(itemDim[0]),"y":int(itemDim[2]),"z":int(itemDim[1])},
        "position" : {"x":int(positions[i][0]),"y":int(positions[i][2]),"z":int(positions[i][1])},
        })
    with open(path,'w') as fp:
        json.dump(data,fp,indent=4)
    print("Done!")
def PossiblePositions(path:str,positions:list[list[int]]):
  data = {
      "points": []
      }
  for pos in positions:
      data["points"].append({
        "position" : {"x":int(pos[0]),"y":int(pos[2]),"z":int(pos[1])},
        })
  with open(path,'w') as fp:
        json.dump(data,fp,indent=4)
