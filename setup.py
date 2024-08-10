from setuptools import setup, Extension
import os
import nanobind

# Define nanobind include directory
nanobind_include_dir = os.path.join(os.path.dirname(nanobind.__file__), 'include')

# Define extension module
ext_modules = [
    Extension(
        'sloth',
        ['sloth.cpp'],
        include_dirs=[
            './lib/Eigen',  # Path to local Eigen directory
            nanobind_include_dir,  # Path to nanobind headers
        ],
        extra_compile_args=['/O2', '/DNDEBUG', '-std=c++22'],
        language='c++'
    ),
]

# Setup configuration
setup(
    name='sloth',
    version='0.0.1',
    author='Evan Porter',
    author_email='evanwporter@gmail.com',
    ext_modules=ext_modules,
    install_requires=['nanobind>=1.0'],  # Ensure nanobind is installed
    zip_safe=False,
)
