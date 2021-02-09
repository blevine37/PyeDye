from distutils.core import setup

setup(name="pyedye",
      version='0.0',
      description='Software for estimating fluorescence quantum yields',
      author='Benjamin G. Levine',
      url='https://github.com/blevine37/PyeDye',
      packages=['pyedye'],
      requires=['numpy','h5py']
      )
