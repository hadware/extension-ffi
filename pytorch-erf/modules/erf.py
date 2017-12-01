from torch.nn.modules.module import Module
from ..functions.batch_erf import BatchERF

class ERF(Module):
    def forward(self, input1, input2):
        return BatchERF()(input1, input2)
