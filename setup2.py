import numpy
from setuptools import setup, find_packages
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import os

if not("LAL_PREFIX" in os.environ):
    print("No LAL installation found, please install LAL from source or source your LAL installation")
    exit()
else:
    lal_prefix = os.environ.get("LAL_PREFIX")

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())
        
    
lal_includes = lal_prefix+"/include"
lal_libs = lal_prefix+"/lib"

ext_modules=[
             Extension("schechter",
                       sources=["schechter_function.pyx"],
                       libraries=["m"], # Unix-like specific
                       include_dirs=[numpy.get_include()]
                       )
             ]

setup(
      name = "schechter",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include()]
      )

ext_modules=[
             Extension("likelihood",
                       sources=["likelihood.pyx"],
                       libraries=["m","lal"], # Unix-like specific
                       library_dirs = [lal_libs],
                       include_dirs=[numpy.get_include(),lal_includes,"./"]
                       )
             ]
setup(
      name = "likelihood",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include(),lal_includes,"./"]
      )

ext_modules=[
             Extension("cosmology",
                       sources=["cosmology.pyx"],
                       libraries=["m","lal"], # Unix-like specific
                       library_dirs = [lal_libs],
                       include_dirs=[numpy.get_include(),lal_includes,"./"]
                       )
             ]
setup(
      name = "cosmology",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include(),lal_includes,"./"]
      )
