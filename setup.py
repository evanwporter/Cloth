from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="wrap_matrix",
        sources=["wrap_matrix.pyx"],
        include_dirs=[np.get_include(), "lib/Eigen"],
        language="c++",
    ),
]

setup(
    name="wrap_matrix",
    ext_modules=cythonize(extensions),
)
