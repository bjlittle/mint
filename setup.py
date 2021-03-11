# -*- python -*-
import setuptools
from pathlib import Path
import os
import glob
import re
import sys


def getCondaVtk():
    # get the VTK installed by conda
    include_dir = Path(sys.exec_prefix) / Path('include')

    try:
        version = list(include_dir.glob('vtk-*'))[-1].name
    except:
        raise RuntimeError('ERROR: you need to "conda install vtk"')

    version = re.sub(r'vtk-', '', version)
    include_dir = str( include_dir / Path(f'vtk-{version}') )
    libraries_dir = str( Path(sys.exec_prefix) / Path('lib') )
    libraries = [lb + f'-{version}' for lb in ['vtkCommonComputationalGeometry', 'vtkIOCore', 
                 'vtkIOLegacy', 'vtkCommonExecutionModel', 'vtkCommonDataModel', 
                 'vtkCommonTransforms', 'vtkCommonMisc', 'vtkCommonMath', 'vtkCommonSystem',
                 'vtkCommonCore', 'vtksys']]
    return {'VTK_VERSION': version,
            'VTK_INCLUDE_DIR': include_dir,
            'VTK_LIBRARIES_DIR': libraries_dir,
            'VTK_LIBRARIES': libraries}


def getCondaNetcdf():
    # get the NetCDF installed by conda
    include_dir = str( Path(sys.exec_prefix) / Path('include') )
    libraries_dir = str( Path(sys.exec_prefix) / Path('lib') )
    libraries = ['netcdf', 'hdf5']

    if not (include_dir / Path('netcdf.h')).exists():
        raise RuntimeError('ERROR: you need to "conda install libnetcdf"')
        
    return {'NETCDF_INCLUDE_DIR': include_dir,
            'NETCDF_LIBRARIES_DIR': libraries_dir,
            'NETCDF_LIBRARIES': libraries}



# extract the MINT version from file version.txt
with open("version.txt") as f:
    VERSION = f.read().strip()


# generate pymint/__init__.py from pymint/__init__.py.in
init_file = ""
with open("pymint/__init__.py.in") as fi:
    init_file = re.sub(r'@VERSION@', VERSION, fi.read())
    with open("pymint/__init__.py", 'w') as fo:
        fo.write(init_file)

vtklib = getCondaVtk()
nclib = getCondaNetcdf()

# C++ 11 flag
cpp11_flag = '-std=c++11'
# give a chance to override the C++ 11 flag
cpp_flags = os.getenv("CPPFLAGS")
if cpp_flags:
    cpp11_flag = cpp_flags # on Windows: '/std:c11'


print(f'VTK_VERSION          = {vtklib["VTK_VERSION"]}')
print(f'VTK_INCLUDE_DIR      = {vtklib["VTK_INCLUDE_DIR"]}')
print(f'VTK_LIBRARIES_DIR    = {vtklib["VTK_LIBRARIES_DIR"]}')
print(f'VTK_LIBRARIES        = {vtklib["VTK_LIBRARIES"]}')
print(f'NETCDF_INCLUDE_DIR   = {nclib["NETCDF_INCLUDE_DIR"]}')
print(f'NETCDF_LIBRARIES_DIR = {nclib["NETCDF_LIBRARIES_DIR"]}')
print(f'NETCDF_LIBRARIES     = {nclib["NETCDF_LIBRARIES"]}')
print(f'C++11 flag           : {cpp11_flag}')


setuptools.setup(
    name="python-mint",
    version=VERSION,
    author="Alexander Pletzer",
    author_email="alexander.pletzer@nesi.org.nz",
    description="Mimetic INterpolation on the sphere",
    long_description="""
Interpolates or regrids an edge centred field from a source grid to either a destination grid or a target element. The 
interpolation is mimetic in the sense that line integrals are conserved from source to destination grids, i.e. Stokes'
theorem is statisfied to near machine precision.
    """,
    long_description_content_type="text/x-rst",
    url="https://github.com/pletzer/mint",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=['mint'],
    ext_modules = [setuptools.Extension('libmint', # name of the shared library
                   sources=glob.glob('src/*.cpp'),
                   define_macros=[],
                   include_dirs=['src/', vtklib["VTK_INCLUDE_DIR"], nclib["NETCDF_INCLUDE_DIR"]],
                   libraries=vtklib["VTK_LIBRARIES"] + nclib["NETCDF_LIBRARIES"],
                   library_dirs=[ vtklib["VTK_INCLUDE_DIR"], nclib["NETCDF_LIBRARIES_DIR"] ],
                   extra_compile_args=[cpp11_flag,],
                   ),],
    include_package_data=True,
    package_dir={'mint': 'pymint'},
    install_requires=['numpy', 'vtk>=8.1.0', 'netcdf4', 'tbb'],
)
