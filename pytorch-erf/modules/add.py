from torch.nn.modules.module import Module
from ..functions.erf import MyAddFunction

class MyAddModule(Module):
    def forward(self, input1, input2):
        return MyAddFunction()(input1, input2)
