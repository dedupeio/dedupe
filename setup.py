from distutils.core import setup, Extension

setup(
      name='Dedupe',
      version='0.3',
      packages=['dedupe','dedupe.test',],
      ext_modules = [Extension("dedupe.affinegap", ["src/affinegap.c"])],
)