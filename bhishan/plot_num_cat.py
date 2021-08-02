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
    "plot_num",
    "plot_cat",
    "plot_num_num",
    "plot_num_cat",
    "plot_cat_num",
    "plot_cat_cat",
    "plot_boxplot_cats_num",
    "plot_count_cat",
    "plot_cat_cat2",
    "plot_num_cat2",
    "plot_cat_binn",
    "plot_cat_cat_pct",
    "plot_donut_binn",
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

# local functions
try:
    from .util_plots import (add_text_barplot, magnify,
                        get_mpl_style, get_plotly_colorscale)
except:
    from util_plots import (add_text_barplot, magnify,
                        get_mpl_style, get_plotly_colorscale)

def plot_num(
    df:DataFrame,
    num:SI,
    xlim:LIMITN=None,
    figsize:LIMIT=(18,12),
    fontsize:int=24,
    xticks:ARRN=None,
    ms:SIN=None,
    line_kws:Dict={},
    bins:str='auto',
    odir:str='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    print_:bool=False,
    disp:bool=False,
    dpi:int=300
    ):
    """Plot numerical column.

    Parameters
    -----------
    df: DataFrame
        DataFrame to plot.
    num: str
        Name of the numerical column.
    xlim: list
        X-axis limit. eg. [0,100]
    figsize: (int,int)
        figure size. e.g. (12,8)
    fontsize: int
        fontsize
    xticks: list
        Xticks for plots.
    ms: int or string
        mpl style name. eg. ggplot, seaborn_darkgrid, -1-3,-100,-300,538,5
    line_kws: dict
        histplot line kwargs
    bins: int
        Number of bins in histogram.
    odir: str
        Name of output directory.
        This directory will be created if it does not exist.
    ofile: str
        Base name of output image. eg. num.png
    save: bool
        Whether to save the image or not.
    show: bool
        Whether or not to show the image.
    print_: bool
        Print output dataframe or not.
    disp: bool
        Display output dataframe or not.
    dpi: int
        Dot per inch saved figure.
    """
    mpl_style = get_mpl_style(ms)
    plt.style.use(mpl_style)
    if not is_numeric_dtype(df[num]):
        raise AttributeError(f'{num} must be a numeric column')

    if disp:
        display(df[num].describe().to_frame().T)

    if print_:
        print(df[num].describe().to_frame().T)

    x = df[num].dropna()

    if xlim:
        x = x[x>=xlim[0]]
        x = x[x<=xlim[1]]

    if not line_kws:
        line_kws={'alpha': 0.2}

    # NOTE: bins is parameter itself in displot, not in kws.

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    sns.histplot(data=x, ax=axes[0][0], line_kws=line_kws,
                    color='blue',bins=bins)
    sns.histplot(data=np.log(x[x>0]), ax=axes[0][1],
                    line_kws=line_kws, color='green',bins=bins)

    sns.boxplot(data=x, ax=axes[1][0],color='purple')
    sns.violinplot(data=x, ax=axes[1][1],color='y')

    # labels
    axes[0][0].set_xlabel(num,fontsize=fontsize)
    axes[0][1].set_xlabel(f'log({num}) (>0)',fontsize=fontsize)
    axes[1][0].set_xlabel(num,fontsize=fontsize)
    axes[1][1].set_xlabel(num,fontsize=fontsize)

    # xticks
    if xticks:
        axes[0][0].set_xticks(xticks)
        axes[1][0].set_xticks(xticks)
        axes[1][1].set_xticks(xticks)

    # title
    axes[0][0].set_title(f'Distribution of **{num}**',fontsize=fontsize)
    axes[0][1].set_title(f'Distribution of **log({num})**',fontsize=fontsize)

    # ticklabels
    for i,j in [[0,0],[0,1],[1,0],[1,1]]:
        axes[i][j].tick_params(axis='x', labelsize=fontsize)
        axes[i][j].tick_params(axis='y', labelsize=fontsize)

    # layout
    plt.tight_layout()

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'{num}.png')

    if save: plt.savefig(ofile, dpi=dpi)
    if show: plt.show(); plt.close()

def plot_cat(
    df:DataFrame,
    cat:SI,
    figsize:LIMIT=(12,8),
    fontsize:int=14,
    odir:str='images',
    ms:SIN=None,
    title:SN=None,
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    print_:bool=False,
    text_kw1:Dict={'comma': True},
    text_kw2:Dict={'percent': True},
    color_palette:str='magma',
    colors:ARRN=None,
    dpi:int=300
    ):
    """Plot the categorical feature.

    Parameters
    -----------
    df: DataFrame
        DataFrame to plot.
    cat: str
        categorical feature to plot.
    figsize: (int,int)
        Figure size.
    fontsize: int
        Fontsize of labels and ticks.
    odir: str
        Name of output directory.
        This directory will be created if it does not exist.
    ms: int or string
        mpl style name. eg. ggplot, seaborn_darkgrid, -1-3,-100,-300,538,5
    ofile: str
        Base name of output image. e.g cat.png
    save: bool
        Whether or not to save the image.
    show: bool
        Whether or not to show the plot.
    print_: bool
        Whether or not to print category counts.
    text_kw1: dict
        Parameter dictionary for add_text.
    text_kw2: dict
        Parameter dictionary for add_text.
    color_palette: str
        Color palette for seaborn plot. e.g. magma
    colors: list
        Colors of categorical columns. e.g ['green','tomato']
    dpi: int
        Dot per inch saved figure.

    """
    # get plot data
    df1 = df[cat].value_counts()
    df1_pct = df1.div(df1.sum()).mul(100)

    # plot attributes
    title = str(title) if title else f"Class distribution of {cat}"
    if colors is not None:
        assert isinstance(colors,list), 'color must be a list'
        color = colors
    if colors is None:
        color = sns.color_palette(color_palette,len(df1))

    plt.style.use(get_mpl_style(ms))
    fig, axes = plt.subplots(1,2,figsize=figsize)

    df1.plot.bar(color=color,ax=axes[0])
    df1_pct.plot.bar(color=color,ax=axes[1])

    add_text_barplot(axes[0],**text_kw1)
    add_text_barplot(axes[1],**text_kw2)

    axes[0].set_xlabel(cat,fontsize=fontsize)
    axes[1].set_xlabel(cat,fontsize=fontsize)
    axes[0].set_ylabel('Count',fontsize=fontsize)
    axes[1].set_ylabel('Percent',fontsize=fontsize)
    plt.subplots_adjust(top=0.72)
    plt.suptitle(title,fontsize=fontsize+2)
    plt.tight_layout()

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'{cat}.png')

    if save: plt.savefig(ofile, dpi=dpi)
    if show: plt.show(); plt.close()

    if print_:
        print('='*50)
        print(f'Feature: **{cat}**')
        print('Overall Count: ')
        for i,v in df1_pct.round(2).items():
            print(f'    {i}: {v}%')

def plot_num_num(
    df:DataFrame,
    num1:SI,
    num2:SI,
    figsize:LIMIT=(12,8),
    fontsize:int=18,
    ms:SIN=None,
    xticks1:ARRN=None,
    xticks2:ARRN=None,
    rot1:int=0,
    rot2:IN=None,
    rot:IN=None,
    odir:str='images',
    line_kws:Dict={},
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Plot the numeric feature.

    Parameters
    -----------
    df: DataFrame
        DataFrame to plot.
    num1: str
        Numerical column 1
    num2: str
        Numerical column 2
    fontsize: int
        Fontsize of x and yticklabels
        (NOT xlabel and ylabels)
    xticks1: list
        xticks for first column.
    xticks2: list
        xticks for second column.
    rot1: int
        Rotation for xticklabels for column 1.
    rot2: int
        Rotation for xticklabels for column 2.
    rot: int
        Rotation for xticks of both columns. (overrides)
    ms: int or string
        mpl style name. eg. ggplot, seaborn_darkgrid, -1-3,-100,-300,538,5
    odir: str
        Name of output directory.
        This directory will be created if it does not exist.
    ofile: str
        Base name of output image. eg. num1_num2.png
    save: bool
        Whether to save the image or not.
    show: bool
        Whether or not to show the image.
    dpi: int
        Dot per inch saved figure.
    """
    mpl_style = get_mpl_style(ms)
    plt.style.use(mpl_style)

    if not is_numeric_dtype(df[num1]):
        raise AttributeError(f'{num1} must be a numeric column')
    if not is_numeric_dtype(df[num2]):
        raise AttributeError(f'{num2} must be a numeric column')

    if not line_kws:
        line_kws ={'alpha': 0.2}

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    sns.histplot(data=df[num1], ax=axes[0][0], line_kws=line_kws,color='blue')
    sns.histplot(data=df[num2], ax=axes[0][1], line_kws=line_kws,color='purple')
    sns.scatterplot(x=num1, y=num2, data=df,ax=axes[1][0],color='cadetblue')
    sns.scatterplot(x=num2, y=num1, data=df,ax=axes[1][1],color='olive')

    # title
    plt.suptitle(f'Scatterplot of {num1} vs {num2}',fontsize=fontsize,y=1.02)

    # labels
    axes[0][0].set_xlabel(num1,fontsize=fontsize)
    axes[0][1].set_xlabel(num2,fontsize=fontsize)
    axes[1][0].set_xlabel(num1,fontsize=fontsize)
    axes[1][1].set_xlabel(num2,fontsize=fontsize)

    axes[0][0].set_ylabel('kde',fontsize=fontsize)
    axes[0][1].set_ylabel('kde',fontsize=fontsize)
    axes[1][0].set_ylabel(num2,fontsize=fontsize)
    axes[1][1].set_ylabel(num1,fontsize=fontsize)

    # xticks1
    if xticks1:
        axes[0][0].set_xticks(xticks1)
        axes[1][0].set_xticks(xticks1)
        axes[1][1].set_yticks(xticks1)

    # xticks2
    if xticks2:
        axes[0][1].set_xticks(xticks2)
        axes[1][0].set_yticks(xticks2)
        axes[1][1].set_xticks(xticks2)

    # ticklabels fontsize
    for i,j in [[0,0],[0,1],[1,0],[1,1]]:
        axes[i][j].tick_params(axis='x', labelsize=fontsize)
        axes[i][j].tick_params(axis='y', labelsize=fontsize)

    # xticklabels1 rotation
    if rot2 == None: rot2 = rot1
    if rot != None: rot1 = rot; rot2 = rot

    plt.setp(axes[0][0].get_xticklabels(), ha="right", rotation=rot1)
    plt.setp(axes[1][0].get_xticklabels(), ha="right", rotation=rot1)

    # xticklabels2 rotation
    plt.setp(axes[0][1].get_xticklabels(), ha="right", rotation=rot2)
    plt.setp(axes[1][1].get_xticklabels(), ha="right", rotation=rot2)

    # layout
    plt.tight_layout()

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'{num1}_{num2}.png')
    if save: plt.savefig(ofile, dpi=dpi)
    if show: plt.show(); plt.close()

def plot_num_cat(
    df:DataFrame,
    num:SI,
    cat:SI,
    figsize:LIMIT=(24,18),
    ms:SIN=None,
    show_hist:bool=False,
    bins:int=100,
    fontsize:int=34,
    odir:str='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    print_:bool=False,
    disp:bool=False,
    dpi:int=300
    ):
    """Plot of continuous variable vs binary-target.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to plot.
    num: str
        Numerical feature which is to be plotted.
    cat: str
        Categorical feature.
    figsize: (int, int)
        Figure size.
    ms: int or string
        mpl style name. eg. ggplot, seaborn_darkgrid, -1-3,-100,-300,538,5
    show_hist: bool
        Choose to show or not histogram plot on top of kdeplot.
    bins: int
        Number of bins in the histogram.
    fontsize: int
        Font size of xlabel and ylabels of all plots.
    odir: str
        Name of output directory.
        This directory will be created if it does not exist.
    ofile: str
        Base name of output image. eg. mycol_target.png
    save: bool
        Whether to save the image or not.
    show: bool
        Whether or not to show the image.
    print_: bool
        Print output dataframe or not.
    disp: bool
        Display output dataframe or not.
    dpi: int
        Dot per inch saved figure.

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        df.bp.plot_num_cat('age','pclass')
    """
    mpl_style = get_mpl_style(ms)
    plt.style.use(mpl_style)
    if not is_numeric_dtype(df[num]):
        raise AttributeError(f'{num} must be a numeric column')

    df = df.dropna(subset=[num])
    unq = sorted(df[cat].dropna().unique())
    fig, ax = plt.subplots(2,2,figsize=figsize)

    # top-left violin and top-right boxplot
    sns.violinplot(x=cat,y=num,data=df,ax=ax[0][0])
    sns.boxplot(x=cat, y=num, data=df,ax=ax[0][1])

    # bottom left kdeplots
    for u in unq:
        ser = df.query(f" {cat} == @u")[num]
        if show_hist:
            sns.histplot(data=ser,label=f"{cat}_{u}", ax=ax[1][0],bins=bins)
        sns.kdeplot(data=ser, label=f"{cat}_{u}", ax=ax[1][0])

    # bottom right stripplot
    sns.stripplot(x=cat,y=num,data=df,ax=ax[1][1])

    # label name
    ax[0][0].set_xlabel(cat,fontsize=fontsize)
    ax[0][0].set_ylabel(num,fontsize=fontsize)
    ax[0][1].set_xlabel(cat,fontsize=fontsize)
    ax[0][1].set_ylabel(num,fontsize=fontsize)
    ax[1][1].set_xlabel(cat,fontsize=fontsize)
    ax[1][1].set_ylabel(num,fontsize=fontsize)
    ax[1][0].set_xlabel(num,fontsize=fontsize)
    ax[1][0].set_ylabel(cat,fontsize=fontsize)

    # ticklabel
    ax[0][0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[0][1].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1][0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1][1].tick_params(axis='both', which='major', labelsize=fontsize)

    # show legend
    ax[1][0].legend(fontsize=fontsize)
    plt.suptitle(f"Plot of {num} vs {cat}",size=fontsize+2, y=1.12)
    plt.tight_layout()

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'{num}_{cat}.png')

    if save: plt.savefig(ofile, dpi=dpi)
    if show: plt.show(); plt.close()

    # display
    if disp:
        display(df.groupby(cat)[num]
        .describe().round(2).T.add_prefix(f'{cat}_').T
        .append(df[num].describe().round(2)))

    if print_:
        print(df.groupby(cat)[num]
        .describe().round(2).T.add_prefix(f'{cat}_').T
        .append(df[num].describe().round(2)))

def plot_cat_num(
    df:DataFrame,
    cat:SI,
    num:SI,
    figsize:LIMIT=(32,24),
    ms:SIN=None,
    add_text:bool=True,
    comma:bool=True,
    decimals:int=2,
    rot:int=75,
    fontsize:int=48,
    odir:str='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False
    ):
    """Plot the categorical feature against numerical feature.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cat: str
        Categorical feature.
    num: str
        Numerical feature.
    figsize: (int,int)
        Figure size. e.g (12,8) gives nice figure for titanic.
    ms: int or string
        mpl style name. eg. ggplot, seaborn_darkgrid, -1-3,-100,-300,538,5
    add_text: bool
        Whether or not to add the text to plots.
    comma: bool
        Whether or not to style the number with comma in text in barplots.
    decimals: int
        Number of decimal points in bar plot text.
    rot: int
        Degree to rotate the text on barplot.
    fontsize: int
        Font size of xlabel and ylabels of all plots.
    odir: str
        Name of output directory.
        This directory will be created if it does not exist.
    ofile: str
        Base name of output image. eg. mycol_target.png
    save: bool
        Whether to save the image or not.
    show: bool
        Whether or not to show the image.

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        df.bp.plot_cat_num('pclass','fare')
    """
    mpl_style = get_mpl_style(ms)
    plt.style.use(mpl_style)
    if not is_numeric_dtype(df[num]):
        raise AttributeError(f'{num} must be a numeric column')

    fig, ax = plt.subplots(3,2,figsize=figsize)

    df1 = df[cat].value_counts(normalize=True).reset_index()
    order = df1['index'].tolist()[::-1]

    pal = 'twilight'

    sns.stripplot(x=cat,y=num, data=df, ax=ax[0][0],order=order,palette=pal)
    sns.violinplot(x=cat,y=num, data=df, ax=ax[0][1],order=order,palette=pal)

    sns.barplot(x=cat,y=num, data=df, ax=ax[1][0],order=order,palette=pal,ci=None)
    (df.groupby(cat)[num].mean() / df.groupby(cat)[num].mean().sum())\
    .plot.bar(color=sns.color_palette(pal,len(order)),ax=ax[1][1])

    sns.countplot(x=df[cat], order=order,palette=pal,ax=ax[2][0])
    sns.barplot(x='index',y=cat, data=df1,order=order,palette=pal,ax=ax[2][1])

    if add_text:
        for i,j in [[1,0],[1,1],[2,0],[2,1]]:
            add_text_barplot(ax[i][j],fontsize=fontsize,comma=comma,
                        decimals=decimals,rot=rot)

    # tick title
    ax[0][0].set_xlabel('',fontsize=fontsize)
    ax[0][1].set_xlabel('',fontsize=fontsize)
    ax[0][0].set_ylabel(num,fontsize=fontsize)
    ax[0][1].set_ylabel('',fontsize=fontsize)

    ax[1][0].set_xlabel('',fontsize=fontsize)
    ax[1][1].set_xlabel('',fontsize=fontsize)
    ax[1][0].set_ylabel(num,fontsize=fontsize)
    ax[1][1].set_ylabel('',fontsize=fontsize)

    ax[2][0].set_xlabel(cat,fontsize=fontsize)
    ax[2][1].set_xlabel(cat,fontsize=fontsize)
    ax[2][0].set_ylabel('count',fontsize=fontsize)
    ax[2][1].set_ylabel('',fontsize=fontsize)

    # ticklabel
    ax[0][0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[0][1].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1][0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1][1].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[2][0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[2][1].tick_params(axis='both', which='major', labelsize=fontsize)

    # remove ticks
    ax[0][0].tick_params(axis='x', which='major', labelsize=0)
    ax[0][1].tick_params(axis='x', which='major', labelsize=0)
    ax[1][0].tick_params(axis='x', which='major', labelsize=0)
    ax[1][1].tick_params(axis='x', which='major', labelsize=0)

    # title
    plt.suptitle(f"Plot of {cat} vs {num}",size=fontsize+2, y=1.12)
    ax[0][0].set_title(f"Swarm plot of {cat} vs {num}",fontsize=fontsize+2)
    ax[0][1].set_title(f"Violin plot of {cat} vs {num}",fontsize=fontsize+2)
    plt.tight_layout()

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'{cat}_{num}.png')

    if save:plt.savefig(ofile)
    if show: plt.show(); plt.close()

def plot_cat_cat(
    df:DataFrame,
    cat:SI,
    ycat:SI,
    figsize:LIMIT=(12,12),
    ms:SIN=None,
    ylim2:ARRN=None,
    rot:int=80,
    fontsize:int=18,
    comma:bool=True,
    loc:str='upper left',
    hide_xticks:bool=False,
    odir:str='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    print_:bool=True
    ):
    """Plot 3*2 plot for categorical feature vs target-categorical feature.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cat: str
        categorical feature.
    ycat: str
        categorical feature for y axis.
    figsize: (int, int)
        Figure size.
    ms: int or string
        mpl style name. eg. ggplot, seaborn_darkgrid, -1-3,-100,-300,538,5
    ylim2: int
        Second plot y-limit upper range.
    rot: int
        Degree to rotate the text on barplot.
    fontsize: int
        Fontsize of xyticks and xylabels.
    comma: bool
        Whether or not to style the number with comma in text in barplots.
    loc: str
        matplotlib plot loc (location) variable. 0 1 2 3 or 'upper right' etc.
    hide_xticks: bool
        Hide xticks of top and middle rows.
    odir: str
        Name of output directory.
        This directory will be created if it does not exist.
    ofile: str
        Base name of output image. eg. mycol_target.png
    save: bool
        Whether to save the image or not.
    show: bool
        Whether or not to show the image.
    print_: bool
        Whether or not to plot the category counts and percents.

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        df.bp.plot_cat_cat('pclass','survived')
    """
    mpl_style = get_mpl_style(ms)
    plt.style.use(mpl_style)

    # data
    rare = df[ycat].value_counts().idxmin()
    df1 = df.groupby([cat,ycat]).count().iloc[:,0].unstack().sort_values(rare)
    order = df1.index.tolist()

    # bottom right
    df1_pct = df1.sum(axis=1).div(df1.sum().sum()).mul(100).round(2)

    # mid right
    df1_pct_cat =  df1.div(df1.sum()).mul(100).round(2)
    df1_pct_cat_sorted = df1_pct_cat.sort_values(rare,ascending=False)
    df1_pct_cat_order = df1_pct_cat.loc[order]

    # top right
    # pd.crosstab(df['BMI_cat'],df['Response8'],normalize='index')
    df1_per_cat = df1.T.div(df1.sum(axis=1)).mul(100).T.round(2)
    df1_per_cat_order = df1_per_cat.loc[order]
    df1_per_cat_sorted = df1_per_cat.sort_values(rare,ascending=False)

    pal = 'magma'
    color=sns.color_palette(pal,df[cat].nunique()) # list of rgb
    ncats = df1.shape[1]
    blues = sns.color_palette("Blues",ncats)
    greens = sns.color_palette("Greens",ncats)

    fig, ax = plt.subplots(3,2,figsize=figsize)

    # top left :
    df1.plot.bar(ax=ax[0][0],color=blues)
    add_text_barplot(ax[0][0],rot=rot,comma=comma)
    ax[0][0].legend(loc=loc)

    # top right:
    df1_per_cat_order.plot.bar(ax=ax[0][1],color=greens)
    add_text_barplot(ax[0][1], percent=True,rot=rot)
    ax[0][1].legend(loc=loc)

    # mid left:  feature vs target count (same as )
    df1.plot.bar(ax=ax[1][0],color=blues)
    add_text_barplot(ax[1][0],rot=rot,comma=comma)
    ax[1][0].legend(loc=loc)

    # mid right percent plot
    df1_pct_cat_order.plot.bar(ax=ax[1][1],color=greens)
    add_text_barplot(ax[1][1], percent=True,rot=rot)
    ax[1][1].legend(loc=loc)

    # bottom left : count plot
    df1.sum(axis=1).rename(ycat).plot.bar(ax=ax[2][0],color=color)
    add_text_barplot(ax[2][0],rot=rot,comma=comma)

    # bottom right : percent plot
    df1_pct.rename(ycat).plot.bar(ax=ax[2][1],color=color)
    add_text_barplot(ax[2][1], percent=True,rot=rot)

    # limits
    if ylim2:
        ax[1][1].set_ylim(0,ylim2)

    # tick title
    ax[0][0].set_xlabel('',fontsize=fontsize)
    ax[0][1].set_xlabel('',fontsize=fontsize)
    ax[0][0].set_ylabel(ycat,fontsize=fontsize)
    ax[0][1].set_ylabel('',fontsize=fontsize)

    ax[1][0].set_xlabel('',fontsize=fontsize)
    ax[1][1].set_xlabel('',fontsize=fontsize)
    ax[1][0].set_ylabel(ycat,fontsize=fontsize)
    ax[1][1].set_ylabel('',fontsize=fontsize)

    ax[2][0].set_xlabel(cat,fontsize=fontsize)
    ax[2][1].set_xlabel(cat,fontsize=fontsize)
    ax[2][0].set_ylabel(ycat,fontsize=fontsize)
    ax[2][1].set_ylabel('',fontsize=fontsize)

    # ticklabel
    ax[0][0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[0][1].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1][0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1][1].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[2][0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[2][1].tick_params(axis='both', which='major', labelsize=fontsize)

    # hide ticks
    if hide_xticks:
        ax[0][0].tick_params(axis='x', which='major', labelsize=0)
        ax[0][1].tick_params(axis='x', which='major', labelsize=0)
        ax[1][0].tick_params(axis='x', which='major', labelsize=0)
        ax[1][1].tick_params(axis='x', which='major', labelsize=0)

    ax[0][0].set_title(f"",fontsize=fontsize+2)
    ax[0][1].set_title(f"",fontsize=fontsize+2)

    # title
    plt.suptitle(f"Plot of {cat} vs {ycat}",size=fontsize+2, y=1.02)
    plt.tight_layout()

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'{cat}_{ycat}.png')

    if save: plt.savefig(ofile)
    if show: plt.show()

    # print
    if print_:
        print('='*50)
        print(f'Feature: **{cat}**')
        print('Overall Count: ')
        for i,v in df[cat].value_counts(ascending=False,normalize=True
                ).mul(100).round(2).items():
            print(f'    {i}: {v}%')

        print()
        print(f'Total  **{ycat}_{rare}** distribution:')
        for i,v in df1_pct_cat_sorted[rare].items():
            print(f'    {i}: {v}%')

        print()
        print(f'Per {cat}  **{ycat}_{rare}** distribution:')
        for i,v in df1_per_cat_sorted[rare].items():
            print(f'    {i}: {v}%')

def plot_cat_cat2(
    df:DataFrame,
    cat:SI,
    target_cat:SI,
    figsize:LIMIT=(12,8),
    ylim2:LIMITN=None,
    ms:SIN=None,
    odir:str='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Plot 2*2 plot for categorical feature vs target-cateogoical feature.

    Parameters
    ----------
    df: pandas.DataFrame
        Input data.
    cat: str
        Categorical feature.
    cat: str
        Categorical target feature.
    figsize: (int, int)
        Figure size.
    ylim2: int
        Upper limit of yaxis.
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
        df.bp.plot_cat_cat('pclass','survived')

    """
    plt.style.use(get_mpl_style(ms))
    fig, ax = plt.subplots(2,2,figsize=figsize)

    # single count plot
    df[cat].value_counts(ascending=True).plot.bar(
    color=sns.color_palette('magma',df[cat].nunique()),ax=ax[0][0])

    # single percent plot
    df[cat].value_counts(ascending=True,normalize=True).mul(100).round(2)\
        .plot.bar(color=sns.color_palette('magma',df[cat].nunique()),
                    ax=ax[0][1])

    # double count  plot
    column0 = pd.crosstab(df[cat],df[target_cat]).columns.min()
    pd.crosstab(df[cat],df[target_cat]).sort_values(column0).plot.bar(ax=ax[1][0])

    # double percent plot
    pd.crosstab(df[cat],df[target_cat],normalize='index')\
    .sort_values(column0).mul(100).round(2).plot.bar(ax=ax[1][1])

    if ylim2:
        ax[1][1].set_ylim(0,ylim2)

    plt.suptitle(f'Count and Percent plot for {cat} vs {target_cat}',
            fontsize=14,color='blue')

    add_text_barplot(ax[0][0])
    add_text_barplot(ax[0][1], percent=True)
    add_text_barplot(ax[1][0])
    add_text_barplot(ax[1][1], percent=True)

    ax[0][0].legend()
    ax[0][1].legend()
    plt.tight_layout()

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'{cat}_vs_{target_cat}.png')

    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show()

def plot_cat_stacked(
    df:DataFrame,
    cols:ARR,
    figsize:LIMIT=(12,8),
    fontsize:int=14,
    ms:SIN=None,
    odir:str='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    kws:Dict={},
    dpi:int=300
    ):
    """Plot stacked plot of categorical features.

    Parameters
    -----------
    cols: list
        List of the categorical column.
    figsize: (int,int)
        figure size. e.g. (12,8)
    fontsize: int
        fontsize
    ms: int or string
        mpl style name. eg. ggplot, seaborn_darkgrid, -1-3,-100,-300,538,5
    odir: str
        Name of output directory.
        This directory will be created if it does not exist.
    ofile: str
        Base name of output image. eg. col.png
    save: bool
        Whether to save the image or not.
    show: bool
        Whether or not to show the image.
    kws: dict
        Kwargs dict for pandas dataframe plot.
        e.g. kws = {'rot':90}
    dpi: int
        Dot per inch saved figure.
    """
    mpl_style = get_mpl_style(ms)
    plt.style.use(mpl_style)

    (df[cols]
    .apply(lambda x: x.value_counts(normalize=True))
    .T
    .plot(kind='bar', stacked=True,figsize=figsize,fontsize=fontsize,**kws)
    )

    # layout
    plt.title('Proportion of values in categorical feature',
            fontsize=fontsize,y=1.05)
    plt.tight_layout()

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'cats_stacked_barchart.png')

    if save: plt.savefig(ofile, dpi=dpi)
    if show: plt.show(); plt.close()

def plot_boxplot_cats_num(
    df:DataFrame,
    cats:ARR,
    num:SI,
    figsize:LIMIT=(12,8),
    ms:SIN=None,
    odir:str='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Plot boxplots in a loop.

    Parameters
    ----------
    df: pandas.DataFrame
        Input data.
    cats: list
        List of categorical columns to plot boxplots.
    num: str
        Name of numerical target column.
    figsize: (int,int)
        Figure size.
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
        df.bp.plot_boxplot_cats_num(['pclass','sex'],'age',show=True)

    """
    plt.style.use(get_mpl_style(ms))
    if not os.path.isdir(odir): os.makedirs(odir)
    for cat in cats:
        plt.figure(figsize=figsize)
        plt.title(f'Box plot of {cat} vs {num}')
        sns.boxplot(x=num, y=cat, data=df,width = 0.8,
                    orient = 'h', showmeans = True, fliersize = 3)

        ofile = os.path.join(odir,f"{cat}_vs_{num}_boxplot.png")
        if save: plt.savefig(ofile,dpi=dpi)
        if show: plt.show(); plt.close()

def plot_count_cat(
    df:DataFrame,
    cat:SI,
    percent:bool=True,
    bottom:int=0,
    figsize:LIMIT=(12,8),
    fontsizex:int=18,
    fontsizey:int=18,
    horizontal:bool=False,
    number:bool=False,
    ms:SIN=None,
    odir:str='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Count plot of given column with optional percent display.

    Useful when just wanting to plot the counts. e.g while doing ML.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cat: str
        Name of categorical column.
    percent: bool
        Whether or not the percent to be shown.
    bottom: float
        Adjust the matplotlib figure bottom.
    figsize: (int, int)
        Figure size.
    fontsizex: int
        Font size for xticks.
    fontsizey: int
        Font size for yticks.
    horizontal: bool
        Whether or not plot horizontal bar plot.
    number: int
        Whether or not to show to number on top of barplots.
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
        df.bp.plot_count_cat(['pclass',show=True)

    """
    plt.style.use(get_mpl_style(ms))
    plt.figure(figsize=figsize)
    total = float(len(df[cat]) )

    if not horizontal:
        ax = sns.countplot(x=cat, data=df, order=df[cat].value_counts().index)
        for p in ax.patches:
            xp = p.get_x()+p.get_width()/2.
            height = p.get_height()
            yp = height
            txt = '{:1.2f}%'.format((height/total)*100)

            if percent:
                txt = '{:d} ({:1.2f}%)'.format(height, (height/total)*100)
            ax.text(xp,bottom,txt, ha="center", fontsize=24, rotation=90,color='black')

    if horizontal:
        ax = sns.countplot(y=cat, data=df, order=df[cat].value_counts().index)
        for p in ax.patches:
            xp = 0.1 # p.get_width() gives 0
            yp = p.get_y()

            # text
            txt_number = str(round(p.get_width(), 2))
            txt_pct = str(round(p.get_width()/total*100, 2)) + '%'

            if number:
                percent = False
                ax.text(xp,yp,txt_number, ha="left", va='top',rotation=0,color='black')

            if percent:
                ax.text(xp,yp,txt_pct, ha="left", va='top',rotation=0,color='black')

    plt.xticks(fontsize=fontsizex,rotation=90)
    plt.yticks(fontsize=fontsizey)
    plt.xlabel(cat, fontsize=fontsizex)
    plt.ylabel('count', fontsize=fontsizey)
    plt.tight_layout()
    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'countplot_{cat}.png')

    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show(); plt.close()

def plot_num_cat2(
    df:DataFrame,
    col_num:SI,
    target_cat:SI,
    figsize:LIMIT=(12,8),
    bins:int=100,
    odir:str='images'
    ):
    """Plot of continuous variable vs binary-target.

    Parameters
    ----------
    df: pandas.DataFrame
        Input data.
    col_num: str
        Name of numeric column.
    target_cat: str
        Name of categorical column.
    figsize: (int, int)
        Figure size.
    bins: int
        Number of bins in the histogram.
    odir: str
        Name of output directory.

    Examples
    ---------
    .. code-block:: python

        col_num = 'age'
        target_cat = 'conversion'
    """

    df = df.dropna(subset=[col_num])
    a,b =  sorted(df[target_cat].dropna().unique())
    fig, ax = plt.subplots(2,2,figsize=figsize)

    # top-left violin and top-right boxplot
    sns.violinplot(x=target_cat,y=col_num,data=df,ax=ax[0][0])
    sns.boxplot(x=target_cat, y=col_num, data=df,ax=ax[0][1])

    # bottom left two-distplots
    ser1 = df.query(f" {target_cat} == @a")[col_num]
    ser2 = df.query(f" {target_cat} == @b")[col_num]
    sns.distplot(ser1,bins=bins,label=f"{target_cat}_{a}", ax=ax[1][0])
    sns.distplot(ser2,bins=bins,label=f"{target_cat}_{b}",ax=ax[1][0])

    # bottom right stripplot
    sns.stripplot(x=target_cat,y=col_num,data=df,ax=ax[1][1])

    # beautify
    ax[1][0].legend()
    plt.tight_layout()

    # save
    plt.savefig(f'{odir}/{col_num}_vs_{target_cat}.png')
    plt.show()

def plot_cat_binn(
    df:DataFrame,
    cat:SI,
    binn:SI,
    zero_one:ARRN=None,
    is_1_good:bool=False,
    names:ARRN=None,
    figsize:LIMIT=(14,5),
    rot:int=0,
    fontsize:int=14,
    palette:str='',
    ofile:str='',
    save:bool=False,
    show:bool=True
    ):
    """Analysis of categorical feature with binary column.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cat: str
        Categorical column
    binn: str
        Binary column.
    zero_one: list
        List of zero and one values of binary column.
        e.g. [0,1], ['no','yes'], ['No','Yes'],['benign','malignant']
    is_1_good: bool
        If 1 means good (eg. Alive) say True.
        Else, if 1 means bad (eg. Churn, Fraud, Malignant) say False.
    names: list
        List of yes and no values of binary column.
    figsize: (int,int)
        Figure size.
    rot: int
        Rotation of text on bar chart.
    fontsize: int
        Size of x and y ticklabels.
        Example:
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis='x',labelsize=fontsize)
    palette: list
        List of color palette names. e.g. ['Reds','Greens']
    ofile: str
        Name of the output file.
    save: bool
        Whether or not to save the image.
    show: bool
        Whether or not to show the image.

    Examples
    ---------
    .. code-block:: python

    df = sns.load_dataset('titanic')
    for col in ['pclass','embarked']:
        plot_cat_binn(df, 'pclass','survived',is_1_good=True)

    """
    # params
    if names is None:
        names = [binn.title(), 'No '+ binn.title()]

    # binary column values
    if zero_one is None:
        # [0,1], ['no','yes'],['No','Yes']
        zero,one = sorted(df[binn].unique())

    # plotting data
    df0 = df[df[binn]==zero]
    df1  = df[df[binn]==one]
    nunique = df[cat].nunique()

    # sanity check
    if nunique >20:
        print(f"Column {cat} has >20 unique values")
        return

    # initiliaze plot
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=figsize)
    plt.subplots_adjust(hspace=0.25)

    def bar_subplot(is_yes=True,palette=palette):
        # palette
        if palette is None:
            if is_1_good:
                palette = 'Greens' if is_yes else 'Reds'
            else:
                palette = 'Reds' if is_yes else 'Greens'
        else:
            palette = palette[0] if is_yes else palette[1]

        # data to plot
        data, plot = [df1, '121'] if is_yes else [df0, '122']
        data_vals = data[cat].value_counts().sort_values()

        # plot attributes
        title = f'{cat} ({names[0]})' if is_yes else f'{cat} ({names[1]})'
        ax = ax1 if is_yes else ax2

        # plot settings
        ax.set_title(title, fontsize=fontsize+4)
        ax.set_xlabel(cat,fontsize=fontsize+4)
        ax.set_ylabel('Count',fontsize=fontsize+4)
        ax.tick_params(axis='x', rotation=rot,labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

        for i, item in enumerate(data_vals):
            txt = "{:,d} ({:0.2f}%)".format(item,item/data_vals.sum()*100)
            ax.text(i, item+data_vals.max()*.018, txt,
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='black')
        sns.countplot(data=data, x=cat,
            palette=palette,
            ax=ax,
            order=data[cat].value_counts(ascending=True).index)

    if not ofile:
        ofile = os.path.join('images',f'{cat}_vs_{binn}.png')
    if save:
        bar_subplot(True)
        bar_subplot(False)
        if not os.path.isdir('images'): os.mkdir('images')
        plt.tight_layout()
        plt.savefig(ofile,dpi=300)
    if show:
        bar_subplot(True)
        bar_subplot(False)
        plt.tight_layout()
        plt.show()
        plt.close()

def plot_cat_cat_pct(
    df:DataFrame,
    col_cat:SI,
    col_cat2:SI,
    figsize:LIMIT=(12,8),
    odir:str=''
    ):
    """Plot of categorical vs binary columns such as class vs survived.

    Parameters
    ----------
    df: pandas.DataFrame
        Input data.
    col_cat: str
        Name of categorical column. e.g. class.
    col_cat2: str
        Name of catetorical column. e.g. survived, embark_town
    figsize: (int, int)
        Figure size.
    odir: str
        Name of output directory.
    """
    x,y = col_cat,col_cat2
    df1 = df.groupby(x)[y].value_counts(normalize=True)
    df1 = df1.mul(100)
    df1 = df1.rename('percent').reset_index()

    g = sns.barplot(x=x,y='percent',hue=y,data=df1)
    g.set_ylim(0,100)

    for p in g.patches:
        txt = str(p.get_height().round(2)) + '%'
        g.text(p.get_x()*1.05,p.get_height()*1.01,txt)

    # beautify
    plt.tight_layout()

    # save
    if odir: plt.savefig(f'{odir}/{col_cat}_vs_{col_cat2}.png')
    plt.show()

def plot_donut_binn(
    df:DataFrame,
    col:SI,
    text:str='',
    colors:LTss=['crimson', 'navy'],
    labels:LTss=['Boring', 'Interesting'],
    figsize:LIMIT=(7,7),
    autopct:str='%1.2f%%'
    ):
    """
    Plot donut plot for a binary column.

    Parameters
    -----------
    df: Pandas DataFrame
        DataFrame object with the data
    col: str
        target column of a binary classification task
    colors: list
        list of two colors.
    labels: list
        list of two labels for 0 and 1 (sorted values)
    autopct: str
        String for percent display in pie plot.

    """
    assert df[col].nunique() == 2, f' {col} Must be a binary feature.'

    if not os.path.isdir('images'):
        os.makedirs('images')

    text = f'Total: \n\n\n{str(len(df))} samples'
    sizes = df[col].value_counts().sort_index().values

    fig, ax = plt.subplots(figsize=figsize)
    center_circle = plt.Circle((0,0), 0.80, color='white')

    ax.pie((sizes[0], sizes[1]), labels=labels, colors=colors, autopct=autopct)
    ax.add_artist(center_circle)
    kwargs = dict(size=20, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **kwargs)
    ax.set_title('Target Class Balance', size=14)
    plt.savefig('images/class_balance_donut_plot.png')
    plt.show()

