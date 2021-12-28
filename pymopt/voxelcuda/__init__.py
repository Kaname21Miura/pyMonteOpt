import os

path = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"


def cuda_init_():
    if os.system("cl.exe"):
        os.environ['PATH'] += ';'+path
    if os.system("cl.exe"):
        raise RuntimeError("cl.exe still not found, path probably incorrect")
cuda_init_()
from ._util import _get_device_config
from ._cukernel import vmc_kernel
from ._classes import VoxelPlateModel
from ._classes import VoxelTuringModel

__all__ = [
'_get_device_config',
'vmc_kernel',

'VoxelPlateModel',
'VoxelTuringModel',
]
