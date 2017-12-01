# functions/add.py
import torch
from torch.autograd import Function
from .._ext import my_lib


class MyAddFunction(Function):
    def forward(self, input_tensor):
        output = input_tensor.new()
        if not input_tensor.is_cuda:
            my_lib.my_lib_add_forward(input_tensor, output)
        else:
            my_lib.my_lib_add_forward_cuda(input_tensor, output)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            my_lib.my_lib_add_backward(grad_output, grad_input)
        else:
            my_lib.my_lib_add_backward_cuda(grad_output, grad_input)
        return grad_input
