from setuptools import setup

setup(name='DisneyDisp',
      version='0.1.1',
      description='A python implementation of the \"Scene Reconstruction from High Spatio-Angular Resolution Light Fields\". For more information see https://www.disneyresearch.com/project/lightfields/.',
      url='https://github.com/manuSrep/DisneyDispPy.git',
      author='Marcel Gutsche, Manuel Tuschen',
      author_email='Manuel_Tuschen@web.de',
      license='GPL3 License',
      packages=['DisneyDisp'],
      install_requires=["scipy","numpy","scikit-image","matplotlib","numba","h5py","progressbar2","easyScripting"],
      zip_safe=False,
      scripts=['clif2lf', 'dmap2hdf5', 'imgs2lf', 'lf2epi', 'disney']
      )
