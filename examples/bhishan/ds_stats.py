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
import numpy as np
import pandas as pd
import scipy

def partial_corr(df,cols,print_=False):
    """Partial correlation coefficent among multiple columns of given array.
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

def point_biserial_correlation(col1, col2):
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

def find_corr(df,cols,target,method='spearman'):
    """Correlation between multiple columns with target column.

    Available methods:
    method : {'pearson', 'kendall', 'spearman'}
    """
    df_corr = (df[cols].corrwith(df[target],method=method)
                    .rename(method+'r')
                    .rename_axis('column')
                    .reset_index())
    return df_corr