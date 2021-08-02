__author__ = 'Bhishan Poudel'

__doc__ = """
This module provides various utilities for statistical analysis.

- partial_corr(df,cols,print_=False)
- point_biserial_correlation(col1, col2)
- find_corr(df,cols,target,method='spearman')

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    help(bp.ds_stats)

"""
__all__ = ["partial_corr","point_biserial_correlation","find_corr"]

# Imports
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
from pandas.core.frame import DataFrame, Series
from .mytyping import (IN, SI, SIN, TL, LD, DS, DSt, NUM, NUMN,
                        AD, AS, DN, ARR, ARRN, SARR, LIMIT, LIMITN,
                        LTii,LTff,LTss,LTsi
                        )

import numpy as np
import pandas as pd
import scipy

def partial_corr(df:DataFrame,cols:ARR,print_:bool=False):
    """Partial correlation coefficient among multiple columns of given array.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe.
    cols: list
        list of feature names
    print_: bool
        Whether or not to print the output dataframe.

    Returns
    --------
    - Pandas Dataframe.
    - if print_ is true, it will also print the output dataframe.

    Usage
    ------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        bp.partial_corr(df,['age','pclass'])

    References
    -------------
    1. https://gist.github.com/fabianp/9396204419c7b638d38f
    """
    arr = df[cols].values
    p = arr.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = scipy.linalg.lstsq(arr[:, idx], arr[:, j])[0]
            beta_j = scipy.linalg.lstsq(arr[:, idx], arr[:, i])[0]
            res_j = arr[:, j] - arr[:, idx].dot( beta_i)
            res_i = arr[:, i] - arr[:, idx].dot(beta_j)
            corr = scipy.stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    df_partial_corr = pd.DataFrame(data=P_corr,columns=cols,index=cols)
    if print_:
        print(df_partial_corr)
    return df_partial_corr

def point_biserial_correlation(col1:SI, col2:SI):
    """Point biserial correlation between two arrays.
    Ref: https://www.kaggle.com/harlfoxem/house-price-prediction-part-1

    Assumptions of point-biserial correlation:
    - There should be no significant outliers in the two groups of the
    dichotomous variable in terms of the continuous variable.
    - There should be homogeneity of variances.
    - The continuous variable should be approximately normally distributed
    for each group of the dichotomous variable.
    """
    r, p = stats.pointbiserialr(df[col1], df[col2])
    print ('Point biserial correlation r is {:.2f} with p = {:.2f}'.format(r,p))

def find_corr(df:DataFrame,cols:ARR,target:SI,method:str='spearman'):
    """Correlation between multiple columns with target column.

    Available methods:
    method : {'pearson', 'kendall', 'spearman'}
    """
    df_corr = (df[cols].corrwith(df[target],method=method)
                    .rename(method+'r')
                    .rename_axis('column')
                    .reset_index())
    return df_corr