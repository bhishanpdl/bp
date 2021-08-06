# bp
[![Build Status](https://travis-ci.org/bhishanpdl/bp.svg?branch=master)](https://travis-ci.org/bhishanpdl/bp)[![Documentation Status](https://readthedocs.org/projects/bp/badge/?version=latest)](https://bp.readthedocs.io/en/latest/?badge=latest)

`bp` is my personal library for day to day use scripts such as data cleaning, machine learning, data visualization and so on.
# API References
For the usage of the module, visit API tutorials at ReadTheDocs or Github Pages.
- [API Usage of bp @ Read the Docs](http://bhishan.readthedocs.io/)
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
Go to your conda environment where you want to install this module, and then install the module.

## Method 01: Install final version
1. First go the folder where is setup.py located.
```bash
ls # there must be setup.py file.
which pip # make sure you are in right environment
python setup.py install --user # This will build some eggs and build directories.
```
- Once we install the module, we can not change the source codes.
- This command will install all the modules given in `setup.py` file.

## Method 02: Install Developer Version
- If we want to change the source code and update the module, we can install developer version.

```bash
which pip # make sure you are in correct conda environment
pip install -e .  # -e means editable version and dot is the path
                  # where there is setup.py file located.

# this will create: bp.egg-info folder.
# make sure you ignore this folder in .gitignore
```

# Using Docker Container
```bash
# step 00: Go to the path where you want to work on
mkdir -p ~/temp; cd ~/temp
mkdir example_docker; cd example_docker

# step 01: download the Dockerfile from this repository
wget https://raw.githubusercontent.com/bhishanpdl/bp/master/Dockerfile
clear;ls
cat Dockerfile

# step 02: Run the docker in your mac (you will see docker icon on menubar)

# step 03: Build the docker image and give name "bp"
# this takes about 2 minutes
# this installs vim, jupyter and some latex modules needed for jupyter-notebook
# --no-cache will delete previous cache
# -t will allow accessing docker image using terminal
docker build --no-cache -t bp .

# step 04: Run the docker image
# allow terminal using -ti
# allow port using -p

docker run -it -p 8888:8888 bp

# if this gives error, try another port
docker run -it -p 8889:8889 bp

# If you have already running this container
# docker ps
# docker stop container_id_obtained_from_docker_ps
# this will open python, do not close it.
# open the new tab on the terminal to go inside the container.

# step 05: Go to docker terminal and run commands there
docker ps
docker exec -ti CONTAINER_ID_from_docker_ps /bin/sh

ls
pwd

# run python script
cd /home/bp/docs/scripts;clear;ls
which python
python --version

python example_json.py

# run jupyter notebook
cd /home/bp/docs/notebooks
# open the second link in the browser, where we can run the notebook.
# hit ctrl c to exit the python
# close the terminal tab where docker image was running.

# step 06: stop and remove the container

# first create new tab on the terminal
# hit ctrl d to close the running python

docker ps
docker stop CONTAINER_ID_from_docker_ps
docker rmi -f bp

Now close the docker app on your machine.
```

# License
This is my personal library intended to be used by only me.
It's not a public library.
