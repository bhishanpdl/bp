__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various utilities to be shared with other modules.

- show_method_attributes(obj, ncols=7,start=None, inside=None,exclude=None,
                            printt=False)

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.utils?

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_method_attributes(obj, ncols=7,start=None, inside=None,exclude=None,
                            printt=False):
    """ Show all the attributes of a given method.

    Parameters
    -----------
    obj: object
        Name of python object. eg. list, pd.DataFrame
    ncols: int
        Number of columns
    start: str
        Substring the attribute starsts with.
    inside: str or tuple or list
        Show only these attributes if given substring exists.
    exclude: str or tuple or list
        Exclude these exact elements
    printt: bool
        Print the dataframe or not.

    """

    # print(f'Object Type: {type(obj)}\n')
    lst = [i for i in dir(obj) if i[0]!='_' ]

    # exclude usual imports
    usual_imports = ['np','pd','os','sys','time','psycopg2']
    lst = [i for i in lst
            if i not in usual_imports ]

    # starts with something
    if isinstance(start,str):
        lst = [i for i in lst if i.startswith(start)]

    if isinstance(start,tuple) or isinstance(start,list):
        lst = [i for i in lst for start_i in start
                if i.startswith(start_i)]

    # inside something
    if isinstance(inside,str):
        lst = [i for i in lst if inside in i]
    if isinstance(inside,tuple) or isinstance(inside,list):
        lst = [i for i in lst for inside_i in inside
                if inside_i in i]

    # exclude substring
    if isinstance(exclude,str):
        lst = [i for i in lst if i != exclude]

    if isinstance(exclude,tuple) or isinstance(exclude,list):
        lst = [i for i in lst if i not in exclude]

    # ouput dataframe
    df = pd.DataFrame(np.array_split(lst,ncols)).T.fillna('')

    # for terminal sometimes we need to print
    if printt:
        print(df)

    return df