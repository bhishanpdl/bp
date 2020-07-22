__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various statistical plotting tools.

- get_yprobs_sorted_proportions(df_ytest,col_ytrue,col_yprobs)
- plot_gini(df_ytest_proportions,col_ytrue,col_yprobs)
- plot_ks(df_ytest_proportions,col_yprobs,
            col_cum_perc_good,col_cum_perc_bad)

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.plot_stats?

"""
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import time
import calendar
from IPython.display import display
from .plot_utils import magnify
from .plot_utils import add_text_barplot

sns.set(color_codes=True)
plt.style.use('ggplot') # better than sns styles.

def get_yprobs_sorted_proportions(df_ytest,col_ytrue,col_yprobs):
    """Sort the df_ytest by predicted probabilities and return
        dataframe with various proportions.

    Parameters:
    ------------
    df_ytest: pd.core.frame.DataFrame
        Test dataframe
    col_ytrue: str
        Name of column for true label.
    col_yprobs: str
        Name of column for predicted probabilities
    """

    df_ytest = df_ytest.sort_values(col_yprobs)
    n_test = len(df_ytest)
    sum_test = df_ytest[col_ytrue].sum()

    df_ytest['cum_n_pop'] = range(1,n_test+1)
    df_ytest['cum_n_good'] = df_ytest['good_bad'].cumsum()
    df_ytest['cum_n_bad'] = (df_ytest['cum_n_pop']
                            - df_ytest['cum_n_good'])

    df_ytest['cum_perc_pop'] = df_ytest['cum_n_pop'] / n_test
    df_ytest['cum_perc_good'] = df_ytest['cum_n_good'] / sum_test
    df_ytest['cum_perc_bad'] = (df_ytest['cum_n_bad']
                                / (n_test - sum_test))

    return df_ytest

def plot_gini(df_ytest_proportions,col_ytrue,col_yprobs):
    """Plot Kolmogorov-Smirnov Curve.

    Parameters:
    ------------
    df_ytest_proportions: pd.core.frame.DataFrame
        Pandas dataframe with at least two columns:
        - cum_perc_pop
        - cum_perc_bad

    Usage:
    -------
    df_ytest = get_yprobs_sorted_proportions(
                    df_ytest,'ytrue','yprobs')

    plot_gini(df_ytest,'ytrue','yprobs')

    """
    from sklearn import metrics
    auc = metrics.roc_auc_score(
        df_ytest_proportions[col_ytrue],
        df_ytest_proportions[col_yprobs])

    gini = 2*auc-1

    x = df_ytest_proportions['cum_perc_pop']
    y = df_ytest_proportions['cum_perc_bad']

    plt.plot(x,y,label=f'Gini = {gini:.4f}')
    plt.plot(x,x,ls='--',c='k')

    plt.xlabel('Cumulative % Population')
    plt.ylabel('Cumulative % Bad')
    plt.title('Gini')

    plt.legend(loc=2)
    plt.show()


def plot_ks(df_ytest_proportions,col_yprobs,
            col_cum_perc_good,col_cum_perc_bad):
    """Plot Kolmogorov-Smirnov Curve.

    Parameters:
    ------------
    df_ytest_proportions: pd.core.frame.DataFrame
        Pandas dataframe with at least three columns:
        - yprob
        - cum_perc_good
        - cum_perc_bad
    col_yprobs: str
        Name of column for test probabilities
    col_cum_perc_good: str
        Name of column for cumulative percent for good
    col_cum_perc_bad: str
        Name of column for cumulative percent for bad

    Usage:
    -------
    df_ytest = get_yprobs_sorted_proportions(
                    df_ytest,'ytrue','yprobs')

    plot_ks(ytest,'yprobs','cum_perc_good','cum_perc_bad')
    """
    x = df_ytest_proportions[col_yprobs]
    y1 = df_ytest_proportions[col_cum_perc_bad]
    y2 = df_ytest_proportions[col_cum_perc_good]

    KS = max(  df_ytest_proportions[col_cum_perc_bad]
            - df_ytest_proportions[col_cum_perc_good]
            )
    KS = round(KS,4)

    plt.plot(x,y1,color='red',label=f'KS = {KS}')
    plt.plot(x,y2,color='blue')

    plt.xlabel('Estimated Probability for being Good')
    plt.ylabel('Cumulative %')
    plt.title('Kolmogorov-Smirnov')

    plt.legend(loc=2)
    plt.show()

