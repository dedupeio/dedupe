from distutils.core import setup, Extension
import numpy

setup(
  name='Dedupe',
  version='0.3',
  packages=['dedupe', 'dedupe.clustering',],
  include_dirs = [numpy.get_include()],
  ext_modules = [Extension("dedupe.affinegap", ["src/affinegap.c"]),
                 Extension("dedupe.lr", ["src/lr.c"]),],
  install_requires = ['numpy', 'fastcluster', 'hcluster']
)
