#include <TH/TH.h>

float erf(float x){
    return
}

int batch_erf_forward(THFloatTensor *input , THFloatTensor *output)
{
    float* input_content = THFloatTensor_data(input);
    int batchSize = (int) input1->size[0];

    return 1;
}

int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
  THFloatTensor_resizeAs(grad_input, grad_output);
  THFloatTensor_fill(grad_input, 1);
  return 1;
}
