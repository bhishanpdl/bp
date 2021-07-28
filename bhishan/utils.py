from __future__ import annotations
__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various utilities to be shared with other modules.

- show_methods(obj, ncols=3,start=None, inside=None,exclude=None,
                            printt=False)

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.utils?
    dir(bp.utils)

"""

__all__ = ['show_methods','parallelize_dataframe','ifnone']


from typing import List,Tuple,Dict,Union,Callable,Any,Sequence,Iterable,Optional
from pandas import DataFrame,Series

# Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp

def ifnone(a,b):
    """ Return if a is None, otherwise return b. """
    return b if a is None else a

def show_methods(obj: Any,
    ncols: int =3,
    starts: str =None,
    contains: str | List | Tuple = None,
    excludes: str | List | Tuple = None,
    exclude_contains: str | List | Tuple = None,
    exclude_starts: str | List | Tuple = None,
    caps_only: bool =False,
    lower_only: bool =False,
    printt: bool=False) -> Dataframe:

    """ Show all the attributes of a given method.

    Parameters
    -----------
    obj: object
        Name of python object. eg. list, pd.DataFrame
    ncols: int
        Number of columns
    starts: str
        Substring the attribute starts with.
    contains: str or tuple or list
        Show only these attributes if given substring exists.
    excludes: str or tuple or list
        Exclude these exact elements
    exclude_contains: str or tuple or list
        Exclude if these appear in elements
    exclude_contains: str or tuple or list
        Exclude if these are the first strings in the elements.
    caps_only: bool
        Show only Title case words
    lower_only: bool
        Show only lowercase case words
    printt: bool
        Print the dataframe or not.

    """

    # print(f'Object Type: {type(obj)}\n')
    lst = [i for i in dir(obj) if i[0]!='_' ]

    # exclude usual imports
    usual_imports = ['np','pd','os','sys','time','psycopg2',
                    'plt','string','px',
                    're','nltk','sklearn','spacy']
    lst = [i for i in lst
            if i not in usual_imports ]

    # capital only (for classes)
    if caps_only:
        lst = [i for i in lst if i[0].isupper()]

    # lowercase only (method attributes)
    if lower_only:
        lst = [i for i in lst if i[0].islower()]

    # starts with something
    if isinstance(starts,str):
        lst = [i for i in lst if i.startswith(starts)]

    if isinstance(starts,tuple) or isinstance(starts,list):
        lst = [i for i in lst for start_i in starts
                if i.startswith(start_i)]

    # inside something
    if isinstance(contains,str):
        lst = [i for i in lst if contains in i]
    if isinstance(contains,tuple) or isinstance(contains,list):
        lst = [i for i in lst for inside_i in contains
                if inside_i in i]

    # exclude exact substring
    if isinstance(excludes,str):
        lst = [i for i in lst if i != excludes]

    if isinstance(excludes,tuple) or isinstance(excludes,list):
        lst = [i for i in lst if i not in excludes]

    # exclude part of substring
    if isinstance(exclude_contains,str):
        lst = [i for i in lst if exclude_contains not in i]

    if isinstance(exclude_contains,tuple) or isinstance(exclude_contains,list):
        lst = [i for i in lst if all(ex not in i for ex in exclude_contains)]

    # exclude if starts
    if isinstance(exclude_starts,str):
        lst = [i for i in lst if not i.startswith(exclude_starts)]

    if isinstance(exclude_starts,tuple) or isinstance(exclude_starts,list):
        lst = [i for i in lst if all(not i.startswith(ex) for ex in exclude_starts)]

    # output dataframe
    df = pd.DataFrame(np.array_split(lst,ncols)).T.fillna('')

    # for terminal sometimes we need to print
    if printt:
        print(df)

    return df

def parallelize_dataframe(df: DataFrame,
    func: Callable) -> DataFrame:
    """ Parallize  df.appy(func) operation to a pandas dataframe.

    Example:
    ========

    def add_text_features(df,colname):
        df = df.copy()
        df['total_length'] = df[colname].apply(len)
        return df

    df = parallelize_dataframe(df, add_text_features)

    """
    ncores = mp.cpu_count()
    df_split = np.array_split(df, ncores)
    pool = mp.Pool(ncores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df