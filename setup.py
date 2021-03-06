#!/usr/bin/env python
# -*- coding: UTF8 -*-
'''
BactoML package
===============

Machine learning tools for the Bactosense project.

:author: Douglas Watson <douglas.watson@bnovate.com>
:date: 2018
:license: Private code, see LICENSE for more information
'''

import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

BASEDIR = os.path.abspath(os.path.dirname(__file__))

long_desc = __doc__
readme_path = os.path.join(BASEDIR, 'README.md')
if os.path.exists(readme_path):
    long_desc = open(readme_path).read()

setup(
    name='BactoML',
    description="Scikit-learn compatible Machine Learning libs for Online Flowcytometry",
    version='0.1',
    author='Marie Sadler, Octave Martin, Eliane Roosli, Ilaria Ricchi, Douglas C. Watson',
    author_email='douglas.watson@bnovate.com',
    platforms=['POSIX'],
    packages=['bactoml'],
    package_data={
        'bactoml': ['testdata/*']
    },
    data_files=[
        # ('/etc/init.d/', ['scripts/whatever']),
    ],
    license='MIT',
    long_description=long_desc,
    install_requires=[
        'numpy',
        'sklearn',
        'FlowCytometryTools',
    ],
    scripts=[
        # 'scripts/example',
    ],
    tests_require=[
        'nose',  # Required for scikit learn's tests
        'pytest',
        'pytest-pep8',
        'pytest-cov',
    ],
)