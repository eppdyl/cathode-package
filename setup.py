import os
from setuptools import setup, find_packages

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('cathode/experimental/files')

setup(name='cathode',
      version='1.0',
      description='Hollow cathode modeling package',
      author='EPPDyL',
      license='None',
      packages=find_packages(),
      ackage_data={'': extra_files},
      install_requires=['numpy', 'scipy'],
      zip_safe=False)
