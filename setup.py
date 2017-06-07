from setuptools import setup,find_packages

setup(name='cathode', 
      version = '0.1',
      description = 'Hollow cathode modeling package',
      author = 'EPPDyL',
      license = 'None',
      packages=find_packages(),
      install_requires=['numpy','scipy'],
      zip_safe = False)
