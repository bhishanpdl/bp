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
    "plot_stem",
    "plot_pareto",
    "countplot",
    "regplot_binn",
    "plot_two_clusters",
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

def plot_stem(
    x:ARR,
    y:ARR,
    label:str='',
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

def plot_pareto(
    df:DataFrame,
    cat:SI,
    value:SIN=None,
    thr:NUMN=None,
    figsize:LIMIT=(12,8),
    rot:int=90,
    fontsize:int=18,
    offset:int=0,
    decimals:int=2,
    ms:SIN=None,
    odir:str='images',
    ofile:str='',
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Pareto Chart.

    Each category must be unique.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cat: str
        Categorical column to get value counts and then plot.
    value: str
        Value column for categorical column of
        already aggregated dataframe.
        If we use grpby all the cat variables must be unique.
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
    decimals: float
        Number of decimals to show in text in barchart.
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
        tips = sns.load_dataset('tips')
        tips.bp.plot_pareto('size',thr=90)

        df = pd.DataFrame({'fruit': ['apple','banana','grape','plum'],
                'price': [17,4,7,7]})
        df.bp.plot_pareto(cat='fruit',value='price',thr=80)

    """
    df_ = self._obj
    plt.style.use(get_mpl_style(ms))
    ylabel = 'Count'
    title = f'Pareto Chart for {cat}'

    if value:
        # Here cat = fruit value = price
        ylabel = value
        title = f'Pareto Chart for {cat}' + f' vs {value}'
        df = df_[[cat,value]].sort_values(value,ascending=False)
        df.index = df[cat].astype(str)
        df["cumperc"] = df[value].cumsum()/df[value].sum()*100

    else:
        # Here, cat = small categorical number or cat
        df = df_[cat].value_counts().to_frame()
        df.index = [str(i) for i in df.index]
        df["cumperc"] = df[cat].cumsum()/df[cat].sum()*100

    df_below = df

    # threshold
    if thr:
        df_below = df[df['cumperc'] < thr].copy()
        cumperc_above = df.loc[df['cumperc'] >= thr, cat].sum()
        if value:
            value_above = df.loc[df['cumperc'] >= thr, value].sum()
            df_below.loc['OTHERS'] = [0] * 3
            df_below.loc['OTHERS',cat] = 0
            df_below.loc['OTHERS',value] = value_above
            df_below.loc['OTHERS','cumperc'] = 100

        else:
            df_below.loc['OTHERS'] = [cumperc_above, 100]

    # plot
    fig, ax1 = plt.subplots(figsize=figsize)

    # barplot
    color = '#C71585'
    if value:
        ax1.bar(df_below.index, df_below[value], color=color)
    else:
        ax1.bar(df_below.index, df_below[cat], color=color)

    # line plot with twinx
    ax2 = ax1.twinx()
    ax2.plot(df_below.index, df_below["cumperc"], color="C1", marker="D", ms=7)

    # 100 % line
    ax2.axhline(y=100,color='r',lw=1,linestyle='--')

    # percent format
    ax2.yaxis.set_major_formatter(PercentFormatter())

    # tick colors
    ax1.tick_params(axis="y", colors=color)
    ax2.tick_params(axis="y", colors="C1")

    # ylim
    ax2.set_ylim([0,135])

    # add text to barplot
    if value:
        for p, (idx,row) in zip(ax1.patches,df_below[[value,'cumperc']].iterrows()) :
            x,y = p.get_x(), p.get_height()
            val,pareto = row[value], row['cumperc']
            pareto = np.round(pareto, decimals=decimals)
            ax1.text(x,y+offset,f'{val:,.0f} ({pareto}%)',
                    fontsize=fontsize,color='blue',
                    rotation=rot,
                    ha='left')
    else:
        for p, (idx,row) in zip(ax1.patches,df_below[[cat,'cumperc']].iterrows()) :
            x,y = p.get_x(), p.get_height()
            val,pareto = row[cat], row['cumperc']
            pareto = np.round(pareto, decimals=decimals)
            ax1.text(x,y+offset,f'{val:,.0f} ({pareto}%)',
                    fontsize=fontsize,color='blue',
                    rotation=rot,
                    ha='left')

    # x and y label
    ax1.set_xlabel(cat,fontsize=fontsize+2)
    ax1.set_ylabel(ylabel,fontsize=fontsize+2)
    ax2.set_ylabel('Pareto',fontsize=fontsize+2)

    # tick fontsize
    ax1.xaxis.set_tick_params(labelsize=fontsize)
    ax1.yaxis.set_tick_params(labelsize=fontsize)
    ax2.yaxis.set_tick_params(labelsize=fontsize)

    # grid
    plt.grid('x')
    plt.tight_layout()
    plt.suptitle(title,fontsize=fontsize)
    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'pareto_chart_{cat}.png')

    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show(); plt.close()

def countplot(
    df:DataFrame,
    cols:ARRN,
    m:int,
    n:int,
    figsize:LIMIT=(12,8),
    fontsize:int=14,
    rot:int=45,
    ms:SIN=None,
    odir:str='images',
    ofile:str='',
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Multiple countplots in a grid of m*n subplots.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list
        List of numerical columns.
    m: int
        Number of plot rows
    n: int
        Number of plot columns.
    figsize: (int,int)
        Figure size.
    fontsize: int
        Size of x and y ticklabels.
    rot: int
        Rotation of text on bar plots.
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
        df.bp.countplot(['pclass','parch'],1,2)

    """
    if cols is None:
        cols = list(df.columns)
    plt.style.use(get_mpl_style(ms))
    fig, ax = plt.subplots(m,n,figsize=figsize)
    for i,col in enumerate(cols):
        plt.subplot(m,n,i+1)
        color=sns.color_palette('husl',df[col].nunique())

        axx = df[col].value_counts(normalize=True).mul(100).plot.bar(color=color)

        plt.xlabel(col,fontsize=fontsize)
        plt.ylim(0,100)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.tight_layout()

        for p in axx.patches:
            txt = str(p.get_height().round(2)) + '%'
            txt_x = p.get_x()
            txt_y = p.get_height()
            plt.text(txt_x,txt_y,txt,fontsize=fontsize-2,rotation=rot)

    for i in range(m*n-len(cols)):
        ax.flat[-(i+1)].set_visible(False)

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'countplots_for_cats.png')

    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show(); plt.close()

def regplot_binn(
    df:DataFrame,
    cols1:ARR,
    cols2:ARR,
    binn:SI,
    m:int,
    n:int,
    figsize:LIMIT=(12,8),
    fontsize:int=18,
    debug:int=False,
    ms:SIN=None,
    odir:str='images',
    ofile:str='',
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Plot multiple lmplots in a grid of m*n subplot.

    Useful to analyze high correlated features.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols1: list
        List of numerical columns.
    cols2: list
        List of numerical columns.
    binn: str
        Binary target feature.
    m: int
        Number of plot rows
    n: int
        Number of plot columns.
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
        df.bp.regplot_binn(['age'], ['fare'],1,2)
        sns.lmplot(x='age',y='fare',data=df,hue='survived')

    """
    plt.style.use(get_mpl_style(ms))
    assert sorted(df[binn].unique().tolist()) == [0,1], 'Binary target must have values 0 and 1'

    msg = f"""\
    sns.lmplot(x='{cols1[0]}',y='{cols2[0]}',data=df,hue='{binn}')
    """
    if debug:
        print(msg)

    fig, axes = plt.subplots(m,n,figsize=figsize)
    for i in range(len(cols1)):
        ax = plt.subplot(m,n,i+1)

        x0 = df.loc[df[binn]==0, cols1[i]]
        y0 = df.loc[df[binn]==0, cols2[i]]
        x1 = df.loc[df[binn]==1, cols1[i]]
        y1 = df.loc[df[binn]==1, cols2[i]]

        sns.regplot(x=x0, y=y0, ax=ax,label='0')
        sns.regplot(x=x1, y=y1, ax=ax,label='1')
        ax.legend(prop={'size':fontsize-2})

        plt.xlabel(cols1[i],fontsize=fontsize)
        plt.ylabel(cols2[i],fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.tight_layout()

    plt.suptitle("Regression plots of Numerical features against Binary target.",
                fontsize=fontsize+4,y=1.05)

    for i in range(m*n-len(cols1)):
        axes.flat[-(i+1)].set_visible(False)

    if ofile:
        # make sure this is base name
        assert ofile == os.path.basename(ofile)
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,ofile)
    else:
        if not os.path.isdir(odir): os.makedirs(odir)
        ofile = os.path.join(odir,f'regplots_of_nums_vs_{binn}.png')

    if save: plt.savefig(ofile,dpi=dpi)
    if show: plt.show(); plt.close()

def plot_two_clusters(
    df:DataFrame,
    cols:ARR,
    target:SI,
    figsize:LIMIT=(24,12),
    fontsize:int=24,
    labels:LTss=['Boring','Interesting']
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