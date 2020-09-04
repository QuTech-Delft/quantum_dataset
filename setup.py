import re

from setuptools import setup
#from setuptools._vendor.packaging.version import Version, InvalidVersion


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def get_version(verbose=1, filename='quantumdataset/version.py'):
    """ Extract version information from source code """

    with open(filename, 'r') as f:
        ln = f.readline()
        m = re.search('.* ''(.*)''', ln)
        version = (m.group(1)).strip('\'')
    if verbose:
        print('get_version: %s' % version)
    return version


setup(name='quantum_dataset',
      version=get_version(),
      use_2to3=False,
      author='Pieter Eendebak',
      author_email='pieter.eendebak@tno.nl',
      maintainer='Pieter Eendebak',
      maintainer_email='pieter.eendebak@tno.nl',
      description='Collection of measurements on quantum devices',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='http://qutech.nl',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering'
      ],
      license='MIT',
      packages=['quantumdataset'],
      install_requires=['numpy', 'qtt', 'MarkupPy', 'imageio', 'PyQt5'],
      tests_require=['pytest'],
      zip_safe=False,
      )
