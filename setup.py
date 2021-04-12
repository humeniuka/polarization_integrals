#!/usr/bin/env python
import os
from os import path
import re
from io import open
from setuptools import setup, find_packages
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
        Pybind11Extension(
                    "polarization_integrals._polarization",
                    sorted(glob("src/*.cc")),
                ),
]

def get_property(property, package):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(property),
        open(package + '/__init__.py').read(),
    )
    return result.group(1)

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.rst'), encoding='utf8') as f:
    long_description = f.read()

setup(
    name='polarization_integrals',
    version=get_property('__version__', 'polarization_integrals'),
    description='Implementation of the polarization integrals according to Schwerdtfeger et al. Phys. Rev. A 37, 2834-2842',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    url='https://github.com/humeniuka/polarization_integrals',
    author='Alexander Humeniuk',
    author_email='alexander.humeniuk@gmail.com',
    license='LICENSE.txt',
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    include_package_data=True,
    zip_safe=False,
    
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)
