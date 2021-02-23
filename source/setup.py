'''
The installation file for the ALENN Bayesian estimation library.

Donovan Platt
Mathematical Institute, University of Oxford
Institute for New Economic Thinking at the Oxford Martin School

2020-12-07

Copyright (c) 2020, University of Oxford. 
All rights reserved.

Distributed under a BSD 3-Clause licence. 
See the accompanying LICENCE file for further details.
'''

#-----------------------------------------------------------------------------
# Load Setup Modules

from setuptools import find_packages, setup

#-----------------------------------------------------------------------------
# Install the Library

setup(name = 'alenn', 
      packages = find_packages(), 
      version = '0.1.0', 
      description = 'ALENN Bayesian estimation library for economic simulation models')