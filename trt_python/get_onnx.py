from turtle import Shape
from typing import KeysView
import torch
import torchvision
torch.cuda.set_device(0)

import os 
import sys
import struct
def get_weights(model_path):
    state_dict=torch.load(model_path,map_location=lambda storage,loc:storage)
    keys=[value for key,value in enumerate(state_dict)]
    weights = dict()
    for key in keys:
        weights[key]=state_dict[key]
    return weights,keys

def extract(weights,keys,weights_path):
    for key in keys:
        print(key)
        value = weights[key]
        Shape = value.shape
        allsize = 1
        for idx in range(len(Shape)):
            allsize*=Shape[idx]
        Value=value.reshape(allsize)
        with open(weights_path+key+'.wgt','wb') as f:
            a=struct.pack('i',allsize)
            f.write(a)
            for i in range(allsize):
                a=struct.pack('f',Value[i])
                f.write(a)
if __name__=='__main__':
    ###save start
    model=torchvision.models.resnet18(pretrained=False)
    print(model)
    torch.save(model.state_dict(),'./model/resnet18.pth')
    model=model.cuda()
    ###save end

    ###export onnx start
    #dummy_input=torch.ones(1,3,244,244,dtype=torch.float32).cuda()
    #torch.onnx.export(model,dummy_input,'./model/resnet18.onnx',verbose=True)
    ###export onnx end

    ###analysis start
    #weights,keys=get_weights('./model/resnet18.pth')
    #extract(weights,keys,'./model/weights/')
    ###analysis end


