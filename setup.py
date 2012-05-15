from distutils.core import setup, Extension


setup(name = 'affinegap',
      version = '0.1',
      ext_modules = [Extension("affinegap", ["affinegap.c"])],
      )
