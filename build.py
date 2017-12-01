import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['pytorch-erf/src/batch_erf.c']
headers = ['pytorch-erf/src/batch_erf.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['pytorch-erf/src/batch_erf_cuda.c']
    headers += ['pytorch-erf/src/batch_erf_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    'pytorch-erf._ext.my_lib',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda
)

if __name__ == '__main__':
    ffi.build()
