from setuptools import find_packages, setup

"""
Updates:

June 02, 2020
Added some functions to sidetable.
The sidetable script was obtained
from python moduel sidetable.

June 03, 2020
Added plot_ds functions to sidetable.

Sep 30, 2020
Added __all__ to all of the .py files.

"""
requirements = ['pandas>=1.0']

setup(
    author='Bhishan Poudel',
    author_email='bhishanpdl@gmail.com',
    name='bhishan',
    version='0.4',
    install_requires=requirements,
    description='Bhishans personal module',
    packages=find_packages(),
    python_requires='>=3.6',
    url='https://github.com/bhishanpdl/bp',
    license='MIT',
)
