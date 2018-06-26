import torch.nn as nn

class RegisterVariable(nn.Module):
    def __init__(self, name, dictOut):
        super(RegisterVariable,self).__init__()        
        print('name',name,'dict',dictOut)
        self.name = name
        self.dictOut = dictOut
    def forward(self,x):
        self.dictOut[self.name]=x
        return x