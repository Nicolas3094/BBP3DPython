import json
import os
def ExpotarJSON(path, cajas:list,bin:list):
    data = {
      "box": [],
      "bin": {
        "x":int(bin[0]),"y":int(bin[2]),"z":int(bin[1])
      }
      }
    for caja in cajas:
        caja = caja
        data["box"].append({
        "id": caja.id,
        "dimension": {"x":int(caja.dimensions[0]),"y":int(caja.dimensions[2]),"z":int(caja.dimensions[1])},
        "position" : {"x":int(caja.MinV[0]),"y":int(caja.MinV[2]),"z":int(caja.MinV[1])},
        })
    with open(path,'w') as fp:
        json.dump(data,fp,indent=4)
def PossiblePositions(path,positions):
  data = {
      "points": []
      }
  for pos in positions:
      data["points"].append({
        "position" : {"x":int(pos[0]),"y":int(pos[2]),"z":int(pos[1])},
        })
  with open(path,'w') as fp:
        json.dump(data,fp,indent=4)
