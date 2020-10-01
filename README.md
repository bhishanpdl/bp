# bp
[![Build Status](https://travis-ci.org/bhishanpdl/bp.svg?branch=master)](https://travis-ci.org/bhishanpdl/bp)[![Documentation Status](https://readthedocs.org/projects/bp/badge/?version=latest)](https://bp.readthedocs.io/en/latest/?badge=latest)

`bp` is my personal library for day to day use scripts such as data cleaning, machine learning, data visualization and so on.
# API References
For the usage of the module, visit API tutorials at ReadTheDocs or Github Pages.
- [API Usage of bp @ Read the Docs](https://bp.readthedocs.io/en/latest/?badge=latest)
- [API Usage of bp @ GitHub Pages](https://bhishanpdl.github.io/bp/)

# Motivation
This repository exists for two reasons.

1. Build a code base for personal use so that I don't have to write the same code snippet twice.
2. Learn the best practices in software industry.

# Examples

**Live Example**  
We can run the `bp` Jupyter notebooks live over the web at [Binder](http://mybinder.org):

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/bhishanpdl/bhishan)

**Rendered Preview**  

|  Notebook | Rendered   | Description  |  Author |
|---|---|---|---|
| demo.ipynb  | [ipynb](https://github.com/bhishanpdl/bp/blob/master/docs/notebooks/demo.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/bp/blob/master/docs/notebooks/demo.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |


# Contributors

* [Bhishan Poudel](https://bhishanpdl.github.io/) (Ph.D Physics, Ohio University)

# Installation
Go to your conda environment and install the module.

## Install final version
```
1. First go the folder where is setup.py located.
which pip # make sure you are in right environment
python setup.py install --user # This will build some eggs and build directories.

# once installed, we can not change the source codes.
# This will installs all the modules given in setup.py file.
```

## Developer Version
```
# If you plan to update the module frequently (I do) use developer method.
which pip # make sure you are in correct conda environment
pip install -e .  # -e means editable version and dot is the path
                  # where there is setup.py file located.

# this will create: bp.egg-info folder.
```

# License
This is my personal library intended to be used by only me.
Its not a public library.
