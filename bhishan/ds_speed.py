__author__ = 'Bhishan Poudel'

__doc__ = """
This module provides various tools and tricks for optimizing the code.

- optimize_memory(df, datetime_features=[None])

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    help(bp.ds_speed)

"""
__all__ = ['optimize_memory']

# Imports
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
from pandas.core.frame import DataFrame, Series
from mytyping import (IN, SN, SI, SIN, TL, LD, DS, DSt, NUM, NUMN,
                        AD, AS, DN, ARR, ARRN, SARR, LIMIT, LIMITN)

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

def optimize_memory(
    df:DataFrame,
    datetime_features:List=[None]
    ):
    """Reduce the memory of pandas dataframe."""
    # How much memory reduced?
    before = df.memory_usage(deep=True).sum() / 1024**2

    # floats
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')

    # ints
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')

    # objects and datetime columns
    for col in df.select_dtypes(include=['object']):
        if col not in datetime_features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_datetime(df[col])

    # How much memory Reduced?
    after = df.memory_usage(deep=True).sum() / 1024**2
    reduction = after - before
    print(f'Memory before  : {before:.2f} MB')
    print(f'Memory after   : {after:.2f} MB')
    print(f'Memory reduced : {reduction:.2f} MB')

    return df