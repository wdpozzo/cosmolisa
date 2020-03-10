import numpy
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

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
                       libraries=["m"], # Unix-like specific
                       include_dirs=[numpy.get_include()]
                       )
             ]
setup(
      name = "likelihood",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include()]
      )
