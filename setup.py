import os
from distutils.core import setup

version = '0.0.1'

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

setup(
    name='pytorch-data-api',
    version=version,
    author='Aleksandr Susha',
    author_email='isushik94@gmail.com',
    description='Data API',
    long_description=README,
    package_dir={'torch_data': 'src'},
    packages=['torch_data'],
    install_requires=INSTALL_REQUIRES,
    keywords=['pytorch', 'torch', 'deep learning', 'data api', 'dataset'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
