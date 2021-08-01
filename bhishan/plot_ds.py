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
    "plot_date_cat",
    "plot_daily_cat",
    "plot_boxplot_cats_num",
    "plot_multiple_jointplots_with_pearsonr",
    "plot_corrplot_with_pearsonr",
    "plot_count_cat",
    "plot_corrplot_with_pearsonr",
    "plot_corr",
    "plot_corr_style",
    "plot_cat_cat2",
    "plot_num_cat2",
    "plot_cat_binn",
    "plot_pareto",
    "plot_cat_cat_pct",
    "plot_donut_binn",
    "plot_two_clusters",
    "plot_stem"
    ]

# Imports
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
from pandas.core.frame import DataFrame, Series
from mytyping import (IN, SN, SI, SIN, TL, LD, DS, DSt, NUM, NUMN,
                        AD, AS, DN, ARR, ARRN, SARR, LIMIT, LIMITN)
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

from plot_utils import magnify
from plot_utils import add_text_barplot
from plot_utils import get_mpl_style
from plot_utils import get_plotly_colorscale

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

SEED = 100

# NOTES: I have most of these functions in pandas api, so I might have
#        slipped some errors. So, find and replace self by df.

def plot_num(
    df:DataFrame,
    col:str,
    xlim:LIMITN=None,
    disp:bool=False,
    print_:bool=False,
    save:bool=False,
    ofile:SN=None,
    show:bool=False):
    """Plot numerical column.

    Parameters
    -----------
    col: str
        Name of the numerical column.
    xlim: list
        X-axis limit. eg. [0,100]
    disp: bool
        Whether or not to display output dataframe.
    print_: bool
        Whether or not to print output dataframe.
    show: bool
        Whether or not to show the plot

    """
    if not is_numeric_dtype(df[col]):
        raise AttributeError(f'{col} must be a numeric column')

    if disp:
        display(df[col].describe().to_frame().T)

    if print_:
        print(df[col].describe().to_frame().T)

    x = df[col].dropna()

    if xlim:
        x = x[x>=xlim[0]]
        x = x[x<=xlim[1]]

    hist_kws={'histtype': 'bar','edgecolor':'black','alpha': 0.2}
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    sns.distplot(x, ax=ax[0][0], hist_kws=hist_kws,
                    color='blue')
    sns.distplot(np.log(x[x>0]), ax=ax[0][1],
                    hist_kws=hist_kws, color='green')

    sns.boxplot(x, ax=ax[1][0],color='purple')
    sns.violinplot(x, ax=ax[1][1],color='y')

    # labels
    ax[0][1].set_xlabel(f'log({col}) (>0)')

    # tight
    plt.tight_layout()

    # save
    if not ofile:
        ofile = f'images/{col}.png'

    if save:
        if not os.path.isdir('images'): os.makedirs('images')
        plt.savefig(ofile)

    # show
    if show:
        plt.show()

def plot_cat(
    df:DataFrame,
    cat:str,
    comma:str=True,
    save:bool=False,
    ofile:SN=None,
    show:bool=False):
    """Plot the categorical feature.

    Parameters
    -----------
    cat: str
        categorical feature to plot.
    comma: bool
        Whether or not to format the number with comma.
    save: bool
        Wheter or not to save the image.
    ofile: str
        Name of output image. e.g images/cat.png
        The folder images is created if it
        does not exist.
    show: bool
        Whether or not to show the plot.

    """
    fig, ax = plt.subplots(1,2)
    df1 = df[cat].value_counts()
    df1_pct = df1.div(df1.sum()).mul(100)

    df1.plot.bar(color=sns.color_palette('magma',len(df1)),ax=ax[0])
    df1_pct.plot.bar(color=sns.color_palette('magma',len(df1)),ax=ax[1])

    add_text_barplot(ax[0],comma=comma)
    add_text_barplot(ax[1],percent=True)
    plt.suptitle(f"Class distribution of {cat}")
    plt.tight_layout()

    if not ofile:
        ofile = f'images/{cat}.png'

    if save:
        if not os.path.isdir('images'): os.makedirs('images')
        plt.savefig(ofile)
    if show:
        plt.show()

    print('='*50)
    print(f'Feature: **{cat}**')
    print('Overall Count: ')
    for i,v in df1_pct.round(2).items():
        print(f'    {i}: {v}%')

def plot_num_num(
    df:DataFrame,
    num1:IS,
    num2:IS,
    figsize:LIMIT=(12,8),
    fontsize:int=12,
    ofile:SN=None,
    save:bool=False,
    show:bool=False):
    """Plot the numeric feature.

    Parameters
    -----------
    num1: str
        Numerical column 1
    num2: str
        Numerical column 2
    fontsize: int
        Fontsize of x and yticklabels
        (NOT xlabel and ylabels)
    ofile: str
        Output file path. eg. images/scatter_a_vs_b.png
    show: bool
        Whether or not to show the image.
    """
    if not is_numeric_dtype(df[num1]):
        raise AttributeError(f'{num1} must be a numeric column')
    if not is_numeric_dtype(df[num2]):
        raise AttributeError(f'{num2} must be a numeric column')

    ax = df.plot.scatter(x=num1,y=num2,
                        figsize=figsize,
                        fontsize=fontsize)
    plt.title(f'Scatterplot of {num1} vs {num2}',fontsize=fontsize)
    plt.xlabel(num1,fontsize=fontsize)
    plt.ylabel(num2,fontsize=fontsize)
    plt.suptitle(f"Plot of {num1} vs {num2}")
    plt.tight_layout()

    if not ofile:
        ofile = f'images/{num1}_vs_{num2}.png'

    if save:
        if not os.path.isdir('images'): os.makedirs('images')
        plt.savefig(ofile)

    if show:
        plt.show()

def plot_num_cat(
    df:DataFrame, 
    num:IS,
    cat:IS,
    figsize:LIMIT=(24,18),
    ms:SIN=None,
    bins:int=100,
    fontsize:int=34,
    odir:str='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    print_:bool=False,
    disp:bool=False):
    """Plot of continuous variable vs binary-target.

    Parameters
    ----------
    df: pandas.DataFrame
        Input data.
    num: str
        Numerical feature which is to be plotted.
    cat: str
        Categorical feature.
    figsize: (int, int)
        Figure size.
    ms: int or string
        mpl style name. eg. ggplot, seaborn_darkgrid, -1-3,-100,-300,538,5
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

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        df.bp.plot_num_cat('age','pclass')

    """
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
        sns.histplot(data=ser,label=f"{cat}_{u}", ax=ax[1][0],bins=bins)
        sns.kdeplot(data=ser,label=f"{cat}_{u}", ax=ax[1][0])

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

    if save: plt.savefig(ofile, dpi=300)
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
    comma:bool=False,
    decimals:int=2,
    rot:int=30,
    add_text:bool=True,
    fontsize:int=14,
    ofile:SI=None,
    save:bool=False,
    show:bool=False):
    """Plot the categorical feature against numerical feature.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cat: str
        Categorical feature.
    num: str
        Numerical feature.
    comma: bool
        Wheter or not to style bar plot with comma.
    decimals: int
        Number of decimal places to use in bar plot text.
    rot: int
        Degree of rotation for text in barplot.
    add_text: bool
        Whether or not to add the text to plots.
    fontsize: int
        Font size of xlabel and ylabels of all plots.
    ofile: str
        Name of output plot. eg. images/mycol_target.png
    save: bool
        Whether to save the image or not.
    show: bool
        Wheter or not to show the image.

    Examples
    ---------
    .. code-block:: python

        categorical feature: age
        target numerical column: salary
    """
    if not is_numeric_dtype(df[num]):
        raise AttributeError(f'{num} must be a numeric column')

    fig, ax = plt.subplots(3,2,figsize=(12,12))

    df1 = df[cat].value_counts(normalize=True).reset_index()
    order = df1['index'].tolist()[::-1]
    pal = np.random.choice(['magma','Paired', 'inferno',
                            'Spectral', 'RdBu', 'BrBG','PRGn',
                            'seismic','bwr','PuOr','twilight'])

    sns.stripplot(x=cat,y=num, data=df, ax=ax[0][0],order=order,palette=pal)
    sns.violinplot(x=cat,y=num, data=df, ax=ax[0][1],order=order,palette=pal)

    sns.barplot(x=cat,y=num, data=df, ax=ax[1][0],order=order,palette=pal)
    (df.groupby(cat)[num].mean() / df.groupby(cat)[num].mean().sum())\
    .plot.bar(color=sns.color_palette(pal,len(order)),ax=ax[1][1])

    sns.countplot(df[cat], order=order,palette=pal,ax=ax[2][0])
    sns.barplot(x='index',y=cat, data=df1,order=order,palette=pal,ax=ax[2][1])

    if add_text:
        add_text_barplot(ax[1][0],comma=comma,decimals=decimals,rot=rot)
        add_text_barplot(ax[1][1],comma=comma,decimals=decimals,rot=rot)
        add_text_barplot(ax[2][0],comma=comma,decimals=decimals,rot=rot)
        add_text_barplot(ax[2][1],comma=comma,decimals=decimals,rot=rot)

    # beautify
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

    plt.suptitle(f"Plot of {cat} vs {num}")
    plt.tight_layout()

    # save
    if not ofile:
        ofile = f'images/{cat}_{num}.png'

    if save:
        if not os.path.isdir('images'): os.makedirs('images')
        plt.savefig(ofile)

    if show:
        plt.show()


def plot_cat_cat(
    df:DataFrame,
    cat:SI,
    ycat:SI,
    figsize:LIMIT=(12,12),
    ms:SIN=None,
    ylim2:LIMITN=None,
    rot:int=80,
    fontsize:int=18,
    text_fontsize:IN=None,
    comma:bool=True,
    loc:SI='upper left',
    hide_xticks:bool=False,
    odir:str='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False,
    print_:bool=True):
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
    text_fontsize: int
        Fontsize of text above bar charts.
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
    if text_fontsize is None:
        text_fontsize= fontsize

    #df = self._obj
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
    add_text_barplot(ax[0][0],rot=rot,comma=comma,fontsize=text_fontsize)
    ax[0][0].legend(loc=loc)

    # top right:
    df1_per_cat_order.plot.bar(ax=ax[0][1],color=greens)
    add_text_barplot(ax[0][1], percent=True,rot=rot,fontsize=text_fontsize)
    ax[0][1].legend(loc=loc)

    # mid left:  feature vs target count (same as )
    df1.plot.bar(ax=ax[1][0],color=blues)
    add_text_barplot(ax[1][0],rot=rot,comma=comma,fontsize=text_fontsize)
    ax[1][0].legend(loc=loc)

    # mid right percent plot
    df1_pct_cat_order.plot.bar(ax=ax[1][1],color=greens)
    add_text_barplot(ax[1][1], percent=True,rot=rot,fontsize=text_fontsize)
    ax[1][1].legend(loc=loc)

    # bottom left : count plot
    df1.sum(axis=1).rename(ycat).plot.bar(ax=ax[2][0],color=color)
    add_text_barplot(ax[2][0],rot=rot,comma=comma,fontsize=text_fontsize)

    # bottom right : percent plot
    df1_pct.rename(ycat).plot.bar(ax=ax[2][1],color=color)
    add_text_barplot(ax[2][1], percent=True,rot=rot,fontsize=text_fontsize)

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
        ofile = os.path.join(odir,f'{cat}_vs_{ycat}.png')

    if save: plt.savefig(ofile)
    if show: plt.show(); plt.close()

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

def plot_cat_stacked(
    df,
    cols,
    figsize=(12,8),
    fontsize=14,
    ms=None,
    odir='images',
    ofile=None,
    save=True,
    show=False,
    kws={}):
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
    """
    # df = self._obj
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

    if save: plt.savefig(ofile, dpi=300)
    if show: plt.show(); plt.close()

#=================================================
def plot_date_cat(
    df:DataFrame,
    col_date:SI,
    target_cat:SI,
    figsize:LIMIT=(8,6),
    show:bool=True,
    save:bool=False):
    """Seasonal plot of datetime column vs target cat.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col_date: str
        datetime feature.
    target_cat: str
        binary target feature
    show: bool
        Whether or not to show the image.
    save: bool
        Whether or not to save the image.

    Usage:
    -------
    df = pd.DataFrame({
        'date': pd.date_range(start='1/1/2018',
                            end='2/1/2019',freq='H'),
        'target': np.random.choice([0,1],size=len(ts))})
    print(df.head(2))

    df.bp.plot_date_cat('date','target',save=True,show=False)

    Note
    -------
    To see full images in jupyter notebook use this:
    .. code-block:: python

        %%javascript
        IPython.OutputArea.auto_scroll_threshold = 9999;

    """
    import calendar

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
    if save:
        if not os.path.isdir('images'): os.mkdir('images')
        ofile = os.path.join('images',f"{col_date}_vs_{target_cat}_dow_count.png")
        plt.savefig(ofile,dpi=300)
    if show:
        plt.show()
        plt.close()

    # month name
    map_month = dict(zip(range(1,13),list(calendar.month_name)[1:]))
    unq = df[col_date].dt.month.unique()
    order = pd.Series(unq).sort_values().map(map_month)
    x = df[col_date].dt.month_name()
    df.groupby([x,target_cat]).count().iloc[:,0].unstack()\
        .loc[order].plot(marker='o',figsize=figsize)
    plt.title(f'Seasonal variation of **{target_cat}** wrt **Month**')
    plt.ylabel('Count')

    if save:
        if not os.path.isdir('images'): os.mkdir('images')
        ofile = os.path.join('images',f"{col_date}_vs_{target_cat}_monthly_count.png")
        plt.savefig(ofile,dpi=300)
    if show:
        plt.show()
        plt.close()

    # day of month
    agg = 'day'
    x =  getattr(df[col_date].dt, agg)
    df.groupby([x,target_cat])\
        .count().iloc[:,0].unstack().plot(marker='o',figsize=figsize)
    plt.title(f'Seasonal variation of **{target_cat}** wrt **Day of Month**')
    plt.xticks(range(x.max()+1))
    plt.ylabel('Count')

    if save:
        if not os.path.isdir('images'): os.mkdir('images')
        ofile = os.path.join('images',f"{col_date}_vs_{target_cat}_daily_count.png")
        plt.savefig(ofile,dpi=300)
    if show:
        plt.show()
        plt.close()

    # hour
    agg = 'hour'
    x = getattr(df[col_date].dt, agg)
    df.groupby([x,target_cat])\
        .count().iloc[:,0].unstack().plot(marker='o',figsize=figsize)
    plt.title(f'Seasonal variation of **{target_cat}** wrt **Hour of Day**')
    plt.xticks(range(x.max()+1))
    plt.ylabel('Count')

    if save:
        if not os.path.isdir('images'): os.mkdir('images')
        ofile = os.path.join('images',f"{col_date}_vs_{target_cat}_hourly_count.png")
        plt.savefig(ofile,dpi=300)
    if show:
        plt.show()
        plt.close()

#===========================================================
def plot_daily_cat(
    df:DataFrame,
    col_date:SI,
    target_cat:SI,
    figsize:LIMIT=(12,8),
    save:bool=False,
    show:bool=False,
    show_xticks:bool=True):
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
    save: bool
        Whether or not to save the image.
    show: bool
        Whether or not to show the image.
    show_xticks: bool
        Whether or not to show the xticks.

    Examples
    ---------
    .. code-block:: python

        col_date = 'date' # datetime format
        target_cat = 'subscribed' # True or False

    NOTE
    -----
    This function plots the daily sum for binary target (eg. True and False).
    """
    if not is_datetime64_any_dtype(df[col_date]):
        raise AttributeError(f'{col_date} must be a datetime column.')

    x = df[col_date].dt.date
    df.groupby([x,target_cat]).count().iloc[:,0].unstack()\
        .plot(marker='o',figsize=figsize)
    xtickvals = pd.date_range(x.min(),x.max(),freq='D')
    plt.xticks(xtickvals, rotation=90)

    if not show_xticks:
        plt.xticks([])

    if save:
        if not os.path.isdir('images'): os.mkdir('images')
        ofile = os.path.join('images',f"{col_date}_vs_{target_cat}_daily.png")
        plt.savefig(ofile,dpi=300)
    if show:
        plt.show()
        plt.close()

def plot_boxplot_cats_num(
    df:DataFrame,
    cats:ARR,
    num:SI,
    save:bool=False,
    show:bool=False):
    """Plot boxplots in a loop.

    Parameters
    ----------
    df: pandas.DataFrame
        Input data.
    cats: list
        List of categorical columns to plot boxplots.
    num: str
        Name of numerical target column.

    Example
    --------
    df = sns.load_dataset('titanic')
    df.bp.plot_boxplot_cats_num(['pclass','sex'],'age',show=True)
    """
    for cat in cats:
        plt.figure(figsize=(12,8))
        plt.title(f'Box plot of {cat} vs {num}')
        sns.boxplot(x=num, y=cat, data=df,
                    width = 0.8,
                orient = 'h', showmeans = True, fliersize = 3)
        if save:
            if not os.path.isdir('images'): os.mkdir('images')
            ofile = os.path.join('images',f"{cat}_vs_{num}_boxplot.png")
            plt.savefig(ofile,dpi=300)
        if show:
            plt.show()
            plt.close()

def plot_multiple_jointplots_with_pearsonr(
    df:DataFrame,
    cols:ARR,
    target:SI,
    ofile:SN):
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

#========================================================
def plot_corrplot_with_pearsonr(
    df:DataFrame,
    cols:ARR,
    ofile:SN=None):
    """Correlation plot with Pearson correlation coefficient.
    Diagonals are distplots, right are scatterplots and left are kde.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list
        List of columns.
    ofile: str
        Name of output image file.

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
    if ofile:
        plt.savefig(ofile, dpi=300)
    plt.show()

#========================================================
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
    save:bool=False,
    show:bool=False):
    """count plot of given column with optional percent display.

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

    """
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
    if save:
        if not os.path.isdir('images'): os.mkdir('images')
        ofile = os.path.join('images',f'countplot_{cat}.png')
        plt.savefig(ofile,dpi=300)
    if show:
        plt.show()
        plt.close()

#=========================================================
def plot_corrplot_with_pearsonr(
    df:DataFrame,
    cols:ARR,
    save:bool=False,
    show:bool=True):
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

#=========================================================
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
    ytitle:NUM=1.05,
    mask:bool=True,
    ms:SIN=None,
    odir:SN='images',
    ofile:SN=None,
    save:bool=True,
    show:bool=False):
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

    if save: plt.savefig(ofile,dpi=300)
    if show:
        plt.show()
        plt.close()

#==========================================================
def plot_corr_style(
    df:DataFrame,
    cols:ARRN=None,
    cmap:str='RdBu'):
    """Correlation plot with style and magnification.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list,optional
        List of columns.
    cmap: str,optional
        Colormap. eg. RdBu, coolward,PuBu

    Examples
    ---------
    .. code-block:: python

        cols = ['sqft_living', 'sqft_living15', 'sqft_above']
        plot_corrplot_with_pearsonr(df,cols)

    """
    if not cols:
        cols = df.columns

    out = df[cols].corr().style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())

    return out

#==========================================================
def plot_cat_cat2(
    df:DataFrame,
    cat:SI,
    target_cat:SI,
    figsize:LIMIT=(12,8),
    ylim2:LIMITN=None,
    ofile:SN=None,
    save:bool=False,
    show:bool=False):
    """Plot 2*2 plot for categorical feature vs target-cateogoical feature.

    Parameters
    ----------
    df: pandas.DataFrame
        Input data.
    cat: str
        Categorical feature.
    target_cat: str
        Categorical target feature.
    figsize: (int, int)
        Figure size.
    ylim2: int
        Upper limit of yaxis.
    ofile: str
        Name of the output file.
    save: bool
        Whether or not to save the image.
    show: bool
        Whether or not to show the image.

    Examples
    ---------
    .. code-block:: python

        cat = 'sex'
        target_cat = 'conversion'
        odir = 'images'

    """
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

    # # double percent plot
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

    if not ofile:
        ofile = os.path.join('images',f'{cat}_vs_{target_cat}.png')
    if save:
        if not os.path.isdir('images'): os.mkdir('images')
        plt.savefig(ofile,dpi=300)
    if show:
        plt.show()
        plt.close()

#===========================================================
def plot_num_cat2(
    df:DataFrame,
    col_num:SI,
    target_cat:SI,
    figsize:LIMIT=(12,8),
    bins:int=100,
    odir:str='images'):
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

#===========================================================
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
    palette:SN=None,
    ofile:SN=None,
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

def plot_pareto(
    df1:DataFrame,
    num:SI,
    cat:SI,
    thr:NUMN=None,
    figsize:LIMIT=(12,8),
    rot:int=90,
    fontsize:int=18,
    offset:int=0,
    decimals:int=2,
    save:bool=False,
    show:bool=False
    ):
    """Pareto Chart.

    Each category must be unique.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    num: str
        Numerical column.
    cat: str
        Categorical column
    thr: int
        Upper threshold for rare categories. e.g 98
    figsize: (int,int)
        Figure size.
    rot: int
        Rotation of text on bar chart.
    yrot: int
        Rotation of ytick labels
    fontsize: int
        Size of x and y ticklabels
    offset: int
        Offset of text above bar chart. eg. 1
    decmals: float
        Number of decimals to show in text in barchart.
    ofile: str
        Name of the output file.
    save: bool
        Whether or not to save the image.
    show: bool
        Whether or not to show the image.

    Examples
    ---------
    .. code-block:: python
        df = pd.DataFrame({
            'price': [ 4.0, 17.0, 7.0, 7.0, 2.0, 1.0, 1.0],
            'item': ['apple', 'banana', 'carrot', 'plum',
                    'orange', 'date', 'cherry']})

        df.bp.plot_pareto('price','item')
        df.bp.plot_pareto('price','item',98)

    """
    df = df1[[cat,num]].sort_values(num,ascending=False)
    df['pareto'] = df[num].cumsum() / df[num].sum() * 100
    df_below = df

    if thr:
        df_below = df[df['pareto'] < thr].copy()
        sum_above = df.loc[df['pareto'] >= thr, num].sum()
        df_below.loc[len(df_below)] = ['OTHERS', sum_above, 100.0]

    fig, axes = plt.subplots(figsize=figsize)

    # label size
    plt.rc('legend',fontsize=20)

    # barplot
    ax1 = df_below.plot(x=cat, y=num,  kind='bar', ax=axes)

    # line plot
    ax2 = df_below.plot(x=cat, y='pareto', marker='D', color="C1",
                kind='line', ax=axes, secondary_y=True)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylim([0,135])

    # add text to barplot
    for p, (idx,row) in zip(ax1.patches,df_below[[num,'pareto']].iterrows()) :
        x,y = p.get_x(), p.get_height()
        value,pareto = row[num], row['pareto']
        pareto = np.round(pareto, decimals=decimals)
        ax1.text(x,y+offset,f'{value:,.0f} ({pareto}%)',
                fontsize=fontsize,color='blue',
                rotation=rot,
                ha='left')
    plt.suptitle(f'Pareto Chart for {num} vs {cat}',fontsize=20)
    ax1.set_xlabel(cat,fontsize=fontsize+2)
    ax1.set_ylabel(num,fontsize=fontsize+2)
    ax2.set_ylabel('Pareto',fontsize=fontsize+2)
    ax2.axhline(y=100,color='r',lw=1,linestyle='--')

    ax1.xaxis.set_tick_params(labelsize=fontsize)
    ax1.yaxis.set_tick_params(labelsize=fontsize)
    ax2.yaxis.set_tick_params(labelsize=fontsize)
    plt.grid('x')
    plt.tight_layout()
    if save:
        if not os.path.isdir('images'): os.mkdir('images')
        ofile = os.path.join('images',f'pareto_chart_{num}_vs_{cat}.png')
        plt.savefig(ofile,dpi=300)
    if show:
        plt.show()
        plt.close()

def plot_cat_cat_pct(
    df:DataFrame,
    col_cat:SI,
    col_cat2:SI,
    figsize:LIMIT=(12,8),
    odir:SN=None
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
    colors:ARR=['crimson', 'navy'],
    labels:ARR=['Boring', 'Interesting'],
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

def plot_two_clusters(
    df:DataFrame,
    cols:ARR,
    target:SI,
    figsize:LIMIT=(24,12),
    fontsize:int=24,
    labels:List[str,str]=['Boring','Interesting']
    ):
    """Plot two clusters using dimensionality reduction methods.

    Parameters
    ----------
    df: pandas.DataFrame
        Input data.
    cols: list
        Name of numeric columns.
    target: str
        Name of binary column.
    figsize: (int, int)
        Figure size.
    fontsize: int
        Font size.
    labels: [str,str]
        Name of the labels.

    """

    X = df[cols].values
    y = df[target].values

    y0,y1 = sorted(df[target].unique())

    # scaling (we must do scaling for pca)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # reduced values
    kwargs = dict(n_components=2,random_state=SEED)
    X_reduced_tsne = TSNE(**kwargs).fit_transform(X)
    X_reduced_pca = PCA(**kwargs).fit_transform(X)
    X_reduced_svd = TruncatedSVD(algorithm='randomized',**kwargs).fit_transform(X)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Two Clusters Visualization Using Dimensionality Reduction', fontsize=fontsize)


    patch0 = mpatches.Patch(color='#0A0AFF', label=labels[0])
    patch1  = mpatches.Patch(color='#AF0000', label=labels[1])
    handles = [patch0, patch1]

    kwargs1 = dict(c=(y == y0),cmap='coolwarm',label=labels[0], linewidths=2)
    kwargs2 = dict(c=(y == y1),cmap='coolwarm',label=labels[1], linewidths=2)

    # t-SNE scatter plot
    ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], **kwargs1)
    ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], **kwargs2)
    ax1.legend(handles=handles,fontsize=fontsize)
    ax1.set_title('t-SNE', fontsize=fontsize)
    ax1.grid(True)

    # PCA scatter plot
    ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], **kwargs1)
    ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], **kwargs2)
    ax2.legend(handles=handles,fontsize=fontsize)
    ax2.set_title('PCA', fontsize=fontsize)
    ax2.grid(True)

    # TruncatedSVD scatter plot
    ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], **kwargs1)
    ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], **kwargs2)
    ax3.legend(handles=handles,fontsize=fontsize)
    ax3.set_title('Truncated SVD', fontsize=fontsize)
    ax3.grid(True)

    ax1.xaxis.set_tick_params(labelsize=fontsize)
    ax1.yaxis.set_tick_params(labelsize=fontsize)
    ax2.xaxis.set_tick_params(labelsize=fontsize)
    ax2.yaxis.set_tick_params(labelsize=fontsize)
    ax3.xaxis.set_tick_params(labelsize=fontsize)
    ax3.yaxis.set_tick_params(labelsize=fontsize)

    if not os.path.isdir('images'):
        os.makedirs('images')

    plt.savefig('images/two_clusters_plot.png',dpi=300)
    plt.show()

def plot_stem(
    x:ARR,
    y:ARR,
    label:SN=None,
    markerfmt:str='x',
    figsize:LIMIT=(8,8),
    color:str="#2ca02c"
    ):
    """Plot Step plot of two variables.

    Parameters
    ----------
    x: array-like
        The x-positions of the stems. Default: (0, 1, ..., len(y) - 1).
    y: array-like
        The y-values of the stem heads.
    label: str, optional, default: None
        The label to use for the stems in legends.
    figsize: (int, int)
        Figure size.
    color: str
        color of markerline and stemline

    Returns
    -------
    container : :class:`~matplotlib.container.StemContainer`
        The container may be treated like a tuple
        (*markerline*, *stemlines*, *baseline*)
    """
    fig,ax = plt.subplots(figsize=figsize)
    m, s, _ = plt.stem(x, y,
                    markerfmt=markerfmt,
                    label=label,
                    use_line_collection=True)
    plt.setp([m, s], color=color)

    if label:
        plt.legend()

    if not os.path.isdir('images'):
        os.makedirs('images')

    plt.savefig('images/stem_plot.png',dpi=300)
    plt.show()

#=============== Plot Helper Functions ==========================
def _annotate_pearsonr(x, y, **kws):
    from scipy import stats
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("pearsonr = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)