from setuptools import setup, Extension
import pybind11

# Define extension module
ext_modules = [
    Extension(
        'sloth',
        ['sloth.cpp'],
        include_dirs=[
            './lib/Eigen',  # Path to local Eigen directory
            pybind11.get_include(),  # Path to pybind11 headers
            pybind11.get_include(user=True)
        ],
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
    install_requires=['pybind11>=2.5'],
    zip_safe=False,
)
