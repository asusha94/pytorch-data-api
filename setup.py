import os
from setuptools import setup

version = '0.1.0'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(ROOT_DIR, 'README.md')) as f:
        README = f.read()
except IOError:
    README = ''

try:
    with open(os.path.join(ROOT_DIR, 'requirements.txt')) as f:
        INSTALL_REQUIRES = [line.strip() for line in f.readlines() if line.strip()]
except IOError:
    INSTALL_REQUIRES = []


# Doesn't work with numpy
def test_suite():
    import sys
    import unittest

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')

    return test_suite


setup(
    name='pytorch-data-api',
    version=version,
    author='Aleksandr Susha',
    author_email='isushik94@gmail.com',
    description='Dataset API for PyTorch',
    long_description=README,
    package_dir={'torch_data': 'src/torch_data',
                 'torch_data._sources': 'src/torch_data/_sources', 'torch_data._ops': 'src/torch_data/_ops'},
    packages=['torch_data', 'torch_data._sources', 'torch_data._ops'],
    test_suite='setup.test_suite',
    install_requires=INSTALL_REQUIRES,
    keywords=['pytorch', 'torch', 'deep learning', 'data api', 'dataset'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
