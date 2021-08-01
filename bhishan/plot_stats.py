__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various statistical plotting tools.

- plot_statistics(df,cols,statistic,color,figsize,ylim)
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
__all__ = [
    "plot_statistics",
    "plot_ecdf",
    "get_yprobs_sorted_proportions",
    "plot_gini",
    "plot_ks"
    ]

# Imports
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
from pandas.core.frame import DataFrame, Series
from mytyping import (IN, SN, SI, SIN, TL, LD, DS, DSt, NUM, NUMN,
                        AD, AS, DN, ARR, ARRN, SARR, LIMIT, LIMITN)

import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt

import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import matplotlib
import json
import os
import time
import calendar
from IPython.display import display
from .plot_utils import magnify
from .plot_utils import add_text_barplot

sns.set(color_codes=True)
plt.style.use('ggplot') # better than sns styles.

def plot_statistics(
    df:DataFrame,
    cols:ARRN=None,
    statistic:str='mean',
    color:str='b',
    figsize:LIMIT=(12,4),
    ylim:LIMITN=None,
    decimals:int=4,
    rot:int=30,
    percent:bool=False,
    comma:bool=False
    ):
    """Plot statistics for given columns.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe.
    cols: list
        list of feature names
    statistic: str
        Name of statistic. e.g. mean,std,skew,kurtosis,median
    color: str
        Color of the plot. e.g. 'red'
    figsize: tuple
        Figure size. e.g. (12,4)
    ylim: tuple
        Y-axis limit to show in the plot.
    decimals: float
        Number of decimal points to show in text.
    rot: int
        Rotation for text
    percent: bool
        Whether or not show percent in text.
    comma: bool
        Whether or not use comma for large numbers in text.

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        df.bp.plot_statistics(['age'],color='b')

    """
    if cols is None:
        cols = df.select_dtypes('number').columns.tolist()

    fig,ax = plt.subplots(figsize=figsize)
    sns.barplot(x=cols, y= df[cols].agg(statistic),color=color,ax=ax)

    if ylim is None:
        ax = add_text_barplot(ax, decimals=4,
                            rot=rot,percent=percent,comma=comma)

    plt.xlabel('Features')
    plt.ylabel(statistic.title())
    plt.title(statistic.title()+ ' of numerical features')

    if ylim:
        plt.ylim(ylim)

    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig(f'images/{statistic}.png',dpi=300)
    plt.show()
    plt.close()

def plot_ecdf(
    df:DataFrame,
    col:SI,
    cross:NUMN=None,
    bins:IN=None,
    color:str='b',
    figsize:LIMIT=(12,4),
    xlim:LIMITN=None,
    ylim:LIMITN=None,
    fontsize:int=12
    ):
    """Plot empirical cumulative distribution function and histogram.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe.
    col: str
        Name of column
    cross: float
        Point at which we plot cross lines for ecdf.
    bins: int
        Number of bins of histogram.
    color: str
        Color of the plot. e.g. 'red'
    figsize: tuple
        Figure size. e.g. (12,4)
    xlim: tuple
        X-axis limit for plots.
    xlim: tuple
        X-axis limit for plots.
    fontsize: int
        Fontsize of the subplot titles.

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        df.bp.plot_ecdf(['age'])

    """
    fig = plt.figure(figsize=(12, 8))
    ax_ecdf = fig.add_subplot(121)
    ax_hist = fig.add_subplot(122)
    ax_hist.set_title('Histogram',fontsize=fontsize)
    ax_hist.set_xlabel(col)
    ax_ecdf.set_xlabel(col)
    ax_hist.set_ylabel("Count",fontsize=fontsize)
    ax_ecdf.set_ylabel("Density",fontsize=fontsize)
    ax_ecdf.set_yticks([i/10 for i in range(11)])

    ax_hist.hist(df[col],bins=bins)

    x, y = np.sort(df[col]), np.arange(1, len(df[col])+1) / len(df[col])

    ax_ecdf.scatter(x, y,color='green')
    ax_ecdf.set_title('Empirical Cumulative Distribution Function',
                    fontsize=fontsize)

    if cross:
        ax_ecdf.axhline(y=cross,linestyle='--',color='orange')
        idx = (np.abs(y - cross)).argmin()
        ax_ecdf.axvline(x[idx],linestyle='--',color='orange')

    plt.suptitle(f"ECDF and Histogram for {col}",fontsize=fontsize+4)

    if xlim is not None:
        ax_hist.set_xlim(xlim)
        ax_ecdf.set_xlim(xlim)

    if ylim is not None:
        ax_hist.set_ylim(ylim)
        ax_ecdf.set_ylim(ylim)

    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig(f'images/{col}_ecdf.png',dpi=300)
    plt.show()
    plt.close()

def get_yprobs_sorted_proportions(
    df_ytest:DataFrame,
    col_ytrue:SI,
    col_yprobs:SI
    ):
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

def plot_gini(
    df_ytest_proportions:DataFrame,
    col_ytrue:SI,
    col_yprobs:SI
    ):
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

def plot_ks(
    df_ytest_proportions:DataFrame,
    col_yprobs:SI,
    col_cum_perc_good:SI,
    col_cum_perc_bad:SI
    ):
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

