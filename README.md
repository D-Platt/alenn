# Approximate Likelihood Estimation using Neural Networks (ALENN)

Donovan Platt
<br>
Mathematical Institute, University of Oxford
<br>
Institute for New Economic Thinking at the Oxford Martin School
<br>
<br>
Copyright (c) 2020, University of Oxford. All rights reserved.
<br>
Distributed under a BSD 3-Clause licence. See the accompanying LICENCE file for further details.

## Overview

A Python implementation of the neural network-based Bayesian estimation methodology introduced by Platt (2021) in the paper *[Bayesian Estimation of Economic Simulation Models Using Neural Networks](https://link.springer.com/article/10.1007/s10614-021-10095-9)*. 
<br>
<br>
While not given a particular name in the original paper, we have retrospectively named the likelihood approximation approach ALENN to provide a compact name by which it may be referred to in future extensions or derivative works.

## Acknowledgements

I would like to thank J. Doyne Farmer, Adrián Carro, and Marco Pangallo for helpful discussions that greatly aided the process of developing this toolbox and the underlying methodology.

## Contents

This toolbox contains a user-friendly and generally-applicable implementation of the ALENN methodology that can easily be employed by users to help solve arbitrary parameter estimation problems for simulation models of their choice. While the methodology itself is generally applicable to both uni- and multivariate data, we have, at this stage, optimised this package for the univariate model output case. A multivariate version is expected to be released in the near future.
<br>
<br>
The toolbox consists of four main folders:
* `source`: contains the primary source code of the toolbox as well as the `setup.py` file required to install the package.
* `environment`: contains a specification of the Python environment in which the toolbox should be utilised.
* `demo`: contains a Jupyter notebook providing a complete tutorial on how to utilise the toolbox to perform Bayesian estimation for simulation models. This is the recommended starting point for new users of the toolbox following the initial installation procedure.
* `replication`: contains a series of Jupyter notebooks that demonstrate how to replicate the primary findings of Section 4 in the paper. This is primarily included for the sake of completeness.

## Installation

Here, we outline the basic steps required to install the toolbox and import it for use within Python. We strongly recommend the use of [Anaconda](https://docs.anaconda.com/) to streamline this process and allow for the best possible user experience.

### Setting-up the Python Environment

As a first step, it is vital to ensure that the toolbox's dependencies and, crucially, the correct versions of these dependencies, are installed. To facilitate this process, we provide, in the `environment` folder, a single file, `alenn_default_environment.yml`, that specifies the Python environment in which ALENN should be run. This file can be easily imported using Anaconda.
<br>
<br>
Of course, users may wish to employ the toolbox within other Python distributions. In such cases, we strongly recommend that users carefully follow the specification presented in `alenn_default_environment.yml` (which can be viewed in most text editors) and in particular install Python 3.7.9, Keras 2.2.4, and Tensorflow 1.15.0 to ensure the correct functioning of the toolbox (Tensorflow 2.0 and later versions of Keras introduce fundamental changes to the respective APIs that are incompatible with this toolbox).

### Installing the Toolbox

Once the Python environment has been correctly set-up, installation is relatively straightforward. Users should set their active directory to be the `source` folder and then run the included `setup.py` file. This is achieved by running (in the Python environment described above) the following terminal command:
```
python setup.py bdist_wheel
```
The above command will create a dist folder containing a wheel file. On our test machine, this wheel file was named: `alenn-0.1.0-py3-none-any.whl`. Once created, the wheel file should be executed (assuming the active directory is still the `source` folder) via the terminal command:
```
pip install dist/alenn-0.1.0-py3-none-any.whl
```

### Importing the Toolbox

Following the above steps, ALENN should now be available for use in Python and can be imported as follows:
```python
import alenn
```
When using ALENN for the first time, we recommend that the `ALENN_Demo.ipynb` notebook in the `demo` folder be consulted for details on how to make use of the installed toolbox.