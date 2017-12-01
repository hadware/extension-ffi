# functions/add.py
import torch
from torch.autograd import Function
from .._ext import batch_erf


class BatchERF(Function):
    derivative_factor = 1.1283791670955126

    def forward(self, input_tensor):
        output = input_tensor.new()
        output.resize_as_(input_tensor)
        if not input_tensor.is_cuda:
            batch_erf.batch_erf_forward(input_tensor, output)
        else:
            batch_erf.batch_erf_forward(input_tensor, output)
        self.save_for_backward(input_tensor)
        return output

    def backward(self, grad_output):
        input_tensor = self.saved_tensors
        derivative = self.derivative_factor * torch.exp(-input_tensor * input_tensor)
        return grad_output * derivative
