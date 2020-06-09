import os
import fnmatch
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as build_py_orig


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('cathode/resources')

setup(name='cathode',
      version='1.0',
      description='Hollow cathode modeling package',
      author='EPPDyL',
      license='AGPLv3',
      package_data={'': extra_files},
      packages=find_packages(),
      install_requires=['numpy', 'scipy','h5py'],
      zip_safe=False)
