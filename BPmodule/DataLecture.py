import json
import os


def ExpotarJSON(path:str,Positions:list[list[int]],cajasID:list[int],Data:list[list[int]],bin:list[int]):
    data = {
      "box": [],
      "bin": {
        "x":int(bin[0]),"y":int(bin[2]),"z":int(bin[1])
      }
      }
    for i in range(len(cajasID)):
        caja = cajasID[i]-1
        data["box"].append({
        "id": caja+1,
        "dimension": {"x":int(Data[caja][0]),"y":int(Data[caja][2]),"z":int(Data[caja][1])},
        "position" : {"x":int(Positions[i][0]),"y":int(Positions[i][2]),"z":int(Positions[i][1])},
        })
    with open(path,'w') as fp:
        json.dump(data,fp,indent=4)
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
