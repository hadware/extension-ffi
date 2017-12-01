#include <iostream>
#include <math.h>
#include <stdio.h>
#include "batch_erf_gpu.h"

extern "C" void batchERF(float* input1, float* output, int batchSize)
{
  int threadPerBlocks = 256;
  int nBlocks = (vocSize + threadPerBlocks - 1) / threadPerBlocks;
  batchPairwiseDistanceKernel<<<nBlocks, threadPerBlocks>>>(input1, input2, output, batchSize, embedDim, vocSize);
}