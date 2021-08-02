__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various utilities for datascience plotting.

I have exported all these functions to pandas api extension.
Use it from there.

From: bp.plot_cat(df,'state')
To  : df.plot_cat('state')

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.plot_ds?

"""
__all__ = [
    "plot_date_cat",
    "plot_daily_cat"
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

# Imports
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
sns.set(color_codes=True)
from tqdm import tqdm_notebook as tqdm
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot') # better than sns styles.
import json
import os
import time
import calendar
from IPython.display import display

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

SEED = 100

# local imports
try:
    from .util_plots import (add_text_barplot, magnify,
                        get_mpl_style, get_plotly_colorscale)
except:
    from util_plots import (add_text_barplot, magnify,
                        get_mpl_style, get_plotly_colorscale)

#================= Timeseries Analysis ========
def plot_date_cat(
    df:DataFrame,
    col_date:SI,
    target_cat:SI,
    figsize:LIMIT=(8,6),
    ms:SIN=None,
    odir:str='images',
    ofile:str='',
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Seasonal plot of datetime column vs target cat.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col_date: str
        datetime feature.
    target_cat: str
        binary target feature
    figsize: (int,int)
        figure size. e.g. (12,8)
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

    Usage:
    -------
    ts = pd.date_range(start='1/1/2018',
                        end='2/1/2019',freq='H')
    target = np.random.choice([0,1],size=len(ts))
    df_ts = pd.DataFrame({'date': ts, 'target': target})
    df_ts.bp.plot_date_cat('date','target',save=True,show=False)

    Note
    -------
    To see full images in jupyter notebook use this:
    .. code-block:: python

        %%javascript
        IPython.OutputArea.auto_scroll_threshold = 9999;

    """
    import calendar
    plt.style.use(get_mpl_style(ms))

    if not is_datetime64_any_dtype(df[col_date]):
        raise AttributeError(f'{col_date} must be a datetime column.')

    # day name
    map_dayofweek = dict(zip(range(7),list(calendar.day_name)))
    unq = df[col_date].dt.dayofweek.unique()
    order = pd.Series(unq).sort_values().map(map_dayofweek)
    x = df[col_date].dt.day_name()
    df.groupby([x,target_cat]).count().iloc[:,0].unstack()\
    .loc[order].plot(marker='o',figsize=figsize)
    plt.title(f'Seasonal variation of **{target_cat}** with **Day of Week**')
    plt.ylabel('Count')
    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f"{col_date}_vs_{target_cat}_dow_count.png")

    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show(); plt.close()

    # month name
    map_month = dict(zip(range(1,13),list(calendar.month_name)[1:]))
    unq = df[col_date].dt.month.unique()
    order = pd.Series(unq).sort_values().map(map_month)
    x = df[col_date].dt.month_name()
    df.groupby([x,target_cat]).count().iloc[:,0].unstack()\
        .loc[order].plot(marker='o',figsize=figsize)
    plt.title(f'Seasonal variation of **{target_cat}** wrt **Month**')
    plt.ylabel('Count')

    ofile = os.path.join(odir,f"{col_date}_vs_{target_cat}_monthly_count.png")
    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show(); plt.close()

    # day of month
    agg = 'day'
    x =  getattr(df[col_date].dt, agg)
    df.groupby([x,target_cat])\
        .count().iloc[:,0].unstack().plot(marker='o',figsize=figsize)
    plt.title(f'Seasonal variation of **{target_cat}** wrt **Day of Month**')
    plt.xticks(range(x.max()+1))
    plt.ylabel('Count')

    ofile = os.path.join(odir,f"{col_date}_vs_{target_cat}_daily_count.png")
    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show(); plt.close()

    # hour
    agg = 'hour'
    x = getattr(df[col_date].dt, agg)
    df.groupby([x,target_cat])\
        .count().iloc[:,0].unstack().plot(marker='o',figsize=figsize)
    plt.title(f'Seasonal variation of **{target_cat}** wrt **Hour of Day**')
    plt.xticks(range(x.max()+1))
    plt.ylabel('Count')

    ofile = os.path.join(odir,f"{col_date}_vs_{target_cat}_hourly_count.png")
    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show(); plt.close()

def plot_daily_cat(
    df:DataFrame,
    col_date:SI,
    target_cat:SI,
    figsize:LIMIT=(12,8),
    show_xticks:bool=True,
    ms:SIN=None,
    odir:str='images',
    ofile:str='',
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Daily total plot of binary target.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col_date: str
        Datetime feature.
    target_cat: str
        Categorical target variable.
    figsize: (int, int)
        Figure size.
    show_xticks: bool
        Whether or not to show the xticks.
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

    ts_small = pd.date_range(start='1/1/2018',
                        end='2/1/2018',freq='H')
    target = np.random.choice([0,1],size=len(ts_small))
    df_ts_small = pd.DataFrame({'date': ts_small, 'target': target})

    df_ts_small.bp.plot_daily_cat('date','target')

    NOTE
    -----
    This function plots the daily sum for categorical target.
    Make sure data frame does not have too may days in x-axis, otherwise
    the plot looks ugly.

    """
    plt.style.use(get_mpl_style(ms))
    if not is_datetime64_any_dtype(df[col_date]):
        raise AttributeError(f'{col_date} must be a datetime column.')

    x = df[col_date].dt.date
    df.groupby([x,target_cat]).count().iloc[:,0].unstack()\
        .plot(marker='o',figsize=figsize)
    xtickvals = pd.date_range(x.min(),x.max(),freq='D')
    plt.xticks(xtickvals, rotation=90)
    plt.tight_layout()

    if not show_xticks:
        plt.xticks([])

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f"{col_date}_vs_{target_cat}_daily.png")

    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show(); plt.close()