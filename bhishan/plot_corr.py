__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various statistical plotting tools.

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.plot_stats?

"""
__all__ = [
    "plot_corr",
    "plot_corr_style",
    "plot_corr_sns",
    "plot_corrplot_with_pearsonr",
    "plot_multiple_jointplots_with_pearsonr"
    ]

# type hints
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
from typing import Optional, Sequence, Type, TypeVar
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame, Series
from pandas.io.formats.style import Styler
try:
    from .mytyping import (IN, SI, SIN, TL, LD, TLN, LDN,
    DS, DSt, NUM, NUMN, AD, AS, DN,
    ARR, ARRN, SARR, SARRN, LIMIT, LIMITN,
    LTii,LTss,LTff,LTsi,
    )
except:
    from mytyping import (IN, SI, SIN, TL, LD, TLN, LDN,
    DS, DSt, NUM, NUMN, AD, AS, DN,
    ARR, ARRN, SARR, SARRN, LIMIT, LIMITN,
    LTii,LTss,LTff,LTsi,
    )

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

# local functions
try:
    from .util_plots import magnify, add_text_barplot
except:
    from util_plots import magnify, add_text_barplot

sns.set(color_codes=True)
plt.style.use('ggplot') # better than sns styles.

# Utility functions
def _annotate_pearsonr(x, y, **kws):
    from scipy import stats
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("pearsonr = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

def plot_corr(
    df:DataFrame,
    cols:ARRN=None,
    target:SIN=None,
    topN:int=10,
    cmap:str='RdYlGn',
    annot:bool=True,
    figsize:LIMIT=(12,8),
    annot_fontsize:int=12,
    xrot:int=0,
    yrot:int=0,
    fontsize:int=18,
    ytitle:int=1.05,
    mask:bool=True,
    ms:SIN=None,
    odir:str='images',
    ofile:str='',
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Correlation plot.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list,optional
        List of columns.
    target: str
        Name of target column
    topN: int
        Top N correlated features with target.
    cmap: str,optional
        Colormap. eg. RdBu, coolwarm,PuBu, RdYlGn
    annot: bool
        Show annotation or not.
    figsize: (int,int)
        Figure size.
    annot_fontsize: int
        Annotation fontsize.
    xrot: int
        Rotation of xtick labels
    yrot: int
        Rotation of ytick labels
    fontsize: int
        Size of x and y ticklabels
    ytitle: float
        Position of title above the figure.
    mask: bool
        Show only upper diagonal values.
    ms: int or string
        mpl style name. eg. ggplot, seaborn_darkgrid, -1-3,-100,-300,538,5
    odir: str
        Name of output directory.
        This directory will be created if it does not exist.
    ofile: str
        Base name of the output image.
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
        df.bp.plot_corr()

        target = 'survived'
        df_few_cols = df[['age','fare']]
        df_few_cols.merge(df[target],left_index=True,right_index=True)\
                .bp.plot_corr()

    """
    plt.style.use(get_mpl_style(ms))
    cols_orig = cols
    if not cols:
        cols = df.columns

    df_corr = df[cols].corr()
    if target:
        colsN = df_corr.nlargest(topN, target).index
        df_corr = df[colsN].corr()

    plt.figure(figsize=figsize)

    if mask:
        with sns.axes_style("white"):
            g = sns.heatmap(df_corr,vmin=-1,vmax=1,cmap=cmap,
                        mask=np.triu(df_corr),
                        annot_kws={'fontsize':annot_fontsize},annot=annot)

    else:
        g = sns.heatmap(df_corr,vmin=-1,vmax=1,cmap=cmap,
                        annot_kws={'fontsize':annot_fontsize},annot=annot)

    g.set_yticklabels(g.get_yticklabels(),
                    rotation=yrot, fontsize=fontsize)
    g.set_xticklabels(g.get_xticklabels(),
                    rotation=xrot, fontsize=fontsize)
    plt.title('Correlation Heatmap',fontsize=fontsize+4,y=ytitle)
    plt.tick_params(axis='both',which='major',labelsize=fontsize,
            labelbottom=True,bottom=False,top=False,labeltop=False)
    plt.tight_layout()

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        name = '_'.join(cols)
        name = 'few_columns' if len(name) > 50 else name
        ofile = os.path.join(odir,f'corrplot_' + name + '.png')

    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show()

def plot_corr_style(
    df:DataFrame,
    cols:ARRN=None,
    target:SIN=None,
    topN:int=10,
    cmap:str='RdBu'
    ):
    """Correlation plot with style and magnification.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list,optional
        List of columns.
    target: str
        Name of target column
    topN: int
        Top N correlated features with target.
    cmap: str,optional
        Colormap. eg. RdBu, coolwarm,PuBu

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        df.bp.plot_corr_style()

    """
    if not cols:
        cols = df.columns

    df_corr = df[cols].corr()
    if target:
        colsN = df_corr.nlargest(topN, target).index
        df_corr = df[colsN].corr()

    out = df_corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())

    return out

def plot_corr_sns(
    df:DataFrame,
    cols:ARRN=None,
    target:SIN=None,
    topN:int=10,
    fontsize:LIMITN=None,
    xrot:int=90,
    yrot:int=0,
    ms:SIN=None,
    odir:str='images',
    ofile:str='',
    save:bool=True,
    show:bool=True,
    dpi:int=300
    ):
    """Correlation plot with Pearson correlation coefficient.
    Diagonals are displots, right are scatterplots and left are kde.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list
        List of columns.
    target: str
        Name of target column
    topN: int
        Top N correlated features with target.
    fontsize: int
        Fontsize for xylabels and ticks
    xrot: int
        x-ticks rotation
    yrot: int
        y-ticks rotation.
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
        df.bp.plot_corr_sns()

    """
    plt.style.use(get_mpl_style(ms))
    if not cols:
        cols = df.columns

    df_corr = df[cols].corr()
    cols = df_corr.columns
    if target:
        colsN = df_corr.nlargest(topN, target).index
        df_corr = df[colsN].corr()
        cols = df_corr.columns

    if not fontsize:
        if len(cols) <= 3:
            fontsize = 14
        else:
            fontsize = len(cols) * 5

    g = sns.PairGrid(data=df_corr, vars = cols, height = 3.5)

    g.map_upper(plt.scatter,color='#8A2BE2')
    g.map_diag(sns.displot)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_lower(_annotate_pearsonr)

    # label rotation
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot,fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=xrot,fontsize=fontsize)
        ax.set_xlabel(ax.get_xlabel(), rotation=xrot,fontsize=fontsize)
        ax.set_ylabel(ax.get_ylabel(), rotation=yrot, fontsize=fontsize,ha='right',va='center')

    # layout
    plt.suptitle('Correlation plot among features',fontsize=fontsize+4,y=1.05)
    plt.tight_layout()

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        name = '_'.join(cols)
        name = 'few_columns' if len(name) > 50 else name
        ofile = os.path.join(odir,f'corr_scatterplot_' + name + '.png')

    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show(); plt.close()

def plot_corrplot_with_pearsonr(
    df:DataFrame,
    cols:ARR,
    save:bool=False,
    show:bool=True
    ):
    """Correlation plot with Pearson correlation coefficient.
    Diagonals are distplots, right are scatterplots and left are kde.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list
        List of columns.
    save: bool
        Whether or not to save the image.
    show: bool
        Whether or not to show the image.

    Examples
    ---------
    .. code-block:: python

        cols = ['sqft_living', 'sqft_living15', 'sqft_above']
        corrplot_with_pearsonr(df,cols)
    """
    g = sns.PairGrid(data=df, vars = cols, height = 3.5)

    g.map_upper(plt.scatter,color='#8A2BE2')
    g.map_diag(sns.distplot)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_lower(_annotate_pearsonr)
    plt.tight_layout()
    if save:
        if not os.path.isdir('images'): os.mkdir('images')
        ofile = os.path.join('images','_'.join(cols) + '_corrplot_pearsonr.png')
        plt.savefig(ofile,dpi=300)
    if show:
        plt.show()
        plt.close()

def plot_multiple_jointplots_with_pearsonr(
    df:DataFrame,
    cols:ARR,
    target:SI,
    ofile:str
    ):
    """Plot multiple jointplots with pearsonr correlation.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list
        List of columns.
    target: str
        Name of target column.
    ofile: str
        Name of the output file.
    """
    import scipy
    for i,col in enumerate(cols):
        p = sns.jointplot(x=col, y=target, data=df, kind = 'reg',
                        height = 5 ,color=colors[i])
        r, _ = scipy.stats.pearsonr(df[col].values, df[target].values)
        p.fig.text(0.33, 0.7, "pearsonr = {:.2f}".format(r),
                ha ='left', fontsize = 15)
        if ofile:
            plt.savefig(ofile, dpi=300)