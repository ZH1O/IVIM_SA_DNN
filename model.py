import torch
import torch.nn as nn

#model
class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.NET = nn.Sequential()
        self.NET.add_module("Linear1",nn.Linear(11, 64))
        self.NET.add_module("elu1",nn.ELU())
        self.NET.add_module("Linear2",nn.Linear(64, 64))
        self.NET.add_module("elu2",nn.ELU())
        self.NET.add_module("Linear3",nn.Linear(64, 64))
        self.NET.add_module("elu3",nn.ELU())
        self.NET.add_module("Linear4",nn.Linear(64, 64))
        self.NET.add_module("elu4",nn.ELU())
        self.NET.add_module("Linear5",nn.Linear(64, 64))
        self.NET.add_module("elu5",nn.ELU())
        self.NET.add_module("Linear6",nn.Linear(64, 64))
        self.NET.add_module("elu6",nn.ELU())
        self.NET.add_module("Linear7",nn.Linear(64, 64))
        self.NET.add_module("elu7",nn.ELU())
        self.NET.add_module("Linear8",nn.Linear(64, 64))
        self.NET.add_module("elu8",nn.ELU())
        self.NET.add_module("Linear9",nn.Linear(64, 64))
        self.NET.add_module("elu9",nn.ELU())
        self.NET.add_module("Linear10",nn.Linear(64, 5))
        self.NET.add_module("elu10",nn.ELU())

    def forward(self, x):
        return torch.abs(self.NET(x))
    
class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        self.NET = nn.Sequential()
        self.NET.add_module("Linear1",nn.Linear(5, 64))
        self.NET.add_module("elu1",nn.ELU())
        self.NET.add_module("Linear2",nn.Linear(64, 64))
        self.NET.add_module("elu2",nn.ELU())
        self.NET.add_module("Linear3",nn.Linear(64, 64))
        self.NET.add_module("elu3",nn.ELU())
        self.NET.add_module("Linear4",nn.Linear(64, 64))
        self.NET.add_module("elu4",nn.ELU())
        self.NET.add_module("Linear5",nn.Linear(64, 64))
        self.NET.add_module("elu5",nn.ELU())
        self.NET.add_module("Linear6",nn.Linear(64, 6))
        self.NET.add_module("elu6",nn.ELU())
    def forward(self, x):
        return torch.abs(self.NET(x))