__author__ = 'Bhishan Poudel'

__doc__ = """
This module helps fitting various data science tools.

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    help(bp.ds_ds)

"""

__all__ = [
    'freq_count',
    'get_column_descriptions',
    'report_cat_binn',
    'compare_kde_binn',
    'compare_kde2'
    ]

# type hints
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
from typing import Optional, Sequence, Type, TypeVar
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame, Series
from pandas.io.formats.style import Styler
try:
    from .mytyping import (IN, SN, SI, SIN, TL, LD, TLN, LDN,
    DS, DSt, NUM, NUMN, AD, AS, DN,
    ARR, ARRN, SARR, SARRN, LIMIT, LIMITN,
    LTii,LTss,LTff,LTsi,
    )
except:
    from mytyping import (IN, SN, SI, SIN, TL, LD, TLN, LDN,
    DS, DSt, NUM, NUMN, AD, AS, DN,
    ARR, ARRN, SARR, SARRN, LIMIT, LIMITN,
    LTii,LTss,LTff,LTsi,
    )

import numpy as np
import pandas as pd
import sklearn
import os
from functools import reduce
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype
import scipy
from scipy import stats
from scipy.stats.morestats import binom_test
import seaborn as sns
sns.set(color_codes=True)
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot') # better than sns styles.
from matplotlib.ticker import PercentFormatter
import matplotlib as mpl
fontsize = 14
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['axes.titlesize'] = fontsize + 2
mpl.rcParams['axes.labelsize'] = fontsize

# local functions
try:
    from .util_plots import (add_text_barplot, magnify,
                        get_mpl_style, get_plotly_colorscale)
except:
    from util_plots import (add_text_barplot, magnify,
                        get_mpl_style, get_plotly_colorscale)

def freq_count(
    df: DataFrame,
    nlargest: Union[int,None]=None
    )-> DataFrame:
    """ Return the item frequency tuple for each unique elements of columns.

    Parameters
    -----------
    df: pandas.DataFrame
        Pandas Dataframe.
    nlargest: int
        Integer to get the n-largest values.

    Usage
    -------
    .. code-block:: python

        cols_cat_small = ['bedrooms', 'bathrooms', 'floors', 'waterfront',
                            'view', 'condition', 'grade']

        dftmp = df[cols_cat_small]
        df_freq = freq_count(dftmp)

    """
    import collections
    vals = df.values.T
    freq_counter = [ collections.Counter(vals[i]).most_common(nlargest)
            for i in range(len(vals)) ]
    df_freq = pd.DataFrame(np.array(freq_counter).T,
                            columns=['item_frequency_tuple'],
                            index=df.columns)
    return df_freq

def get_column_descriptions(
    df: DataFrame,
    column_list: Union[Iterable,None]=None,
    style:bool=False
    )->DSt:
    """Get nice table of columns description of given dataframe.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe.
    column_list: list
        list of feature names
    style: bool
        Whether to render the dataframe with style or not.

    Usage
    ------
    .. code-block:: python

        from bhishan.util_ds import get_column_descriptions
        get_column_descriptions(df)

    """
    if column_list is None:
        column_list = df.columns

    df_desc = pd.DataFrame()
    df_desc['column'] = df.columns
    df_desc['dtype'] = df.dtypes.values
    df_desc['nans'] = df.isnull().sum().values
    df_desc['nzeros'] = df.eq(0).sum().values
    df_desc['nunique'] = df.nunique().values

    df_desc['nans_pct'] = df_desc['nans'].div(len(df)).mul(100).round(2).values
    df_desc['nzeros_pct'] = df_desc['nzeros'].div(len(df)).mul(100).round(2).values

    df_desc['nans_pct'] = df_desc['nans_pct'].astype(str) + '%'
    df_desc['nzeros_pct'] = df_desc['nzeros_pct'].astype(str) + '%'
    df_desc = df_desc[['column', 'dtype', 'nunique',
                        'nans', 'nans_pct','nzeros','nzeros_pct']]
    if style:
        df_desc_styled = (df_desc.style
                            .apply(lambda x: ["background: salmon"
                            if  v == 'object' else "" for v in x], axis = 1)
                            )
        df_desc = df_desc_styled

    return df_desc

def report_cat_binn(
    df: DataFrame,
    cat: SI,
    binn: SI,
    one: SI,
    name: SI
    )->None:
    """Analysis of categorical feature with binary column.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cat: str
        Categorical column
    binn: str
        Binary column
    one: str, int
        Value of binary feature corresponding to 1. e.g. 'Yes'
    name: str
        Name of binary feature. e.g. Churn, Fraud, Alive

    Examples
    ---------
    .. code-block:: python

    df = sns.load_dataset('titanic')
    report_cat_binn(df,'pclass','survived', one=1,name='Alive')
    """
    total = df.shape[0]

    dfx = df[cat].value_counts().reset_index()
    dfx['index'] = cat + '_' + dfx['index'].astype(str)

    cat_types = list(dfx['index'])
    cat_values = list(dfx.iloc[:,1])

    cat_types_pct = df[cat].value_counts(
        normalize=True).mul(100).values

    cat_yes_values = pd.crosstab(df[binn],
                                df[cat]).loc[one].values

    cat_yes_pct_group = (pd.crosstab(df[binn], df[cat],
                normalize='columns'
                ).mul(100).loc[one].values)

    cat_yes_pct_totalyes = pd.crosstab(df[binn], df[cat],
                normalize='index'
                ).mul(100).loc[one].values
    max_len = max([len(i) for  i in cat_types])
    empty_len = max_len + len(name) + 11
    print()
    print(f'Feature: {cat}')
    print('='*50)
    for i in range(len(cat_types)):
        print(f"{cat_types[i]:{max_len}s} : {cat_values[i]:<5d}"
                f" ({cat_types_pct[i]:5.2f}% of total {total:<5d})")
    print()
    empty=" "
    for i in range(len(cat_types)):
        print(f"""\
{name}_{cat_types[i]:{max_len}s} : {cat_yes_values[i]:<5d} (\
{cat_yes_pct_totalyes[i]:5.2f}% of {cat_yes_values.sum():<5d} total {name} and
{empty:{empty_len}s}{cat_yes_pct_group[i]:5.2f}% of {cat_values[i]:<5d} group {cat_types[i]}\
)""")

def compare_kde_binn(
    df:DataFrame,
    cols:ARR,
    binn:ARR,
    m:int=1,
    n:int=1,
    bw_adjust:NUM=0.5,
    figsize:LIMIT=(12,8),
    fontsize:int=14,
    loc:str='best',
    ms:SIN=None,
    odir:str='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Compare the KDE plots of two numerical features against binary target.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list
        List of numerical columns.
    binn: str
        Binary target feature.
    m: int
        Number of plot rows
    n: int
        Number of plot columns.
    bw_adjust: float
        Bandwidth of kde plot.
    figsize: (int,int)
        Figure size.
    fontsize: int
        Size of x and y ticklabels.
    loc: str or int
        Location of legend. eg. 'best', 'lower left', 'upper left'
    ms: int or string
        mpl style name. eg. ggplot, seaborn_darkgrid, -1-3,-100,-300,538,5
    odir: str
        Name of output directory.
        This directory will be created if it does not exist.
    ofile: str
        Base name of output image.
    save: bool
        Whether or not to save the image.
    show: bool
        Whether or not to show the image.
    dpi: int
        Dot per inch saved figure.

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        df.bp.compare_kde(['age','fare'],'survived',1,2)

    """
    if isinstance(cols,str) or isinstance(cols,int):
        cols = [cols]
    df = df[cols+[binn]].dropna(how='any')
    plt.style.use(get_mpl_style(ms))
    assert sorted(df[binn].unique().tolist()) == [0,1], 'Binary target must have values 0 and 1'

    t0 = df.loc[df[binn] == 0]
    t1 = df.loc[df[binn] == 1]

    fig, ax = plt.subplots(m,n,figsize=figsize)

    for i,col in enumerate(cols):
        plt.subplot(m,n,i+1)
        sns.kdeplot(t0[col], bw_adjust=bw_adjust,label=f"{binn} = 0")
        sns.kdeplot(t1[col], bw_adjust=bw_adjust,label=f"{binn} = 1")
        plt.ylabel('')
        plt.xlabel(col, fontsize=fontsize)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.legend(prop=dict(size=fontsize),loc=loc)

    # remove empty subplots
    for i in range(m*n-len(cols)):
        ax.flat[-(i+1)].set_visible(False)

    # title
    plt.title(f'Compare kde plots')
    plt.tight_layout()

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'compare_kde_cats_vs_{binn}.png')

    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show(); plt.close()

def compare_kde2(
    df:DataFrame,
    num:SI,
    binn:SI,
    figsize:LIMIT=(12,8),
    fontsize:int=14,
    ms:SIN=None,
    odir:str='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Compare the KDE plots of two numerical features against binary target.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    num: str
        Numerical feature.
    binn: str
        Binary feature.
    figsize: (int,int)
        Figure size.
    fontsize: int
        Size of x and y ticklabels.
    ms: int or string
        mpl style name. eg. ggplot, seaborn_darkgrid, -1-3,-100,-300,538,5
    odir: str
        Name of output directory.
        This directory will be created if it does not exist.
    ofile: str
        Base name of output image.
    save: bool
        Whether or not to save the image.
    show: bool
        Whether or not to show the image.
    dpi: int
        Dot per inch saved figure.

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        df.bp.compare_kde('fare','survived')

    References
    -----------

    `stackoverflow <https://stackoverflow.com/questions/62375034/find-non-overlapping-area-between-two-kde-plots-in-python>`_
    """
    note = """I assume this is not working. When I compared it with sns.kde
            with binary target "Response8" and "risk_Age_medium_bool"
            variable of Prudential Life Insurance data, it gave me
            bimodal distribution, but clearly it is unimodel.
            So this function does not work properly.
    """
    if note:
        return "Currently not available."

    df = df[[num,binn]].dropna(how='any')
    plt.style.use(get_mpl_style(ms))
    assert sorted(df[binn].unique().tolist()) == [0,1], 'Binary target must have values 0 and 1'

    x0 = df.loc[df[binn] == 0, num]
    x1 = df.loc[df[binn] == 1, num]

    kde0 = stats.gaussian_kde(x0, bw_method=0.3)
    kde1 = stats.gaussian_kde(x1, bw_method=0.3)

    xmin = min(x0.min(), x1.min())
    xmax = min(x0.max(), x1.max())
    dx = 0.2 * (xmax - xmin)    # add a 20% margin,
                                # as the kde is wider than the data
    xmin -= dx
    xmax += dx

    x = np.linspace(xmin, xmax, 500)
    kde0_x = kde0(x)
    kde1_x = kde1(x)
    inters_x = np.minimum(kde0_x, kde1_x)

    plt.plot(x, kde0_x, color='b', label='No')
    plt.fill_between(x, kde0_x, 0, color='b', alpha=0.2)
    plt.plot(x, kde1_x, color='orange', label='Yes')
    plt.fill_between(x, kde1_x, 0, color='orange', alpha=0.2)
    plt.plot(x, inters_x, color='r')
    plt.fill_between(x, inters_x, 0, facecolor='none',
                    edgecolor='r', hatch='xx', label='intersection')

    area_inters_x = np.trapz(inters_x, x)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels[2] += f': {area_inters_x * 100:.1f} %'
    plt.legend(handles, labels, title=binn)
    plt.title(f'{num} vs {binn}')
    plt.tight_layout()

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'compare_kde_{num}_vs_{binn}.png')

    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show(); plt.close()