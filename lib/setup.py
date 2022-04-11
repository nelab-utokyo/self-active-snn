
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("utils_c", ["utils_c.pyx"],
        include_dirs=[numpy.get_include()]),
]
setup(
    ext_modules = cythonize(extensions)
)

