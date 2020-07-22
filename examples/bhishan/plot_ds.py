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
from .plot_utils import magnify
from .plot_utils import add_text_barplot


def plot_num(df,col,xlim=None,disp=False,print_=False,save=False,ofile=None,show=False):
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

    hist_kws={'histtype': 'bar',
                'edgecolor':'black',
                'alpha': 0.2}

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


def plot_cat(df,cat,comma=True,save=False,ofile=None,show=False):
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

def plot_num_num(df,num1,num2,figsize=(12,8),fontsize=12,ofile=None,save=False,show=False):
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
        ofile = f'images/{num1}_{num2}.png'

    if save:
        if not os.path.isdir('images'): os.makedirs('images')
        plt.savefig(ofile)

    if show:
        plt.show()


def plot_num_cat(self, col_num,target_cat,
                figsize=(12,8),bins=100,fontsize=14,
                ofile=None,save=None,disp=False,
                print_=False,show=False):
    """Plot of continuous variable vs binary-target.

    Parameters
    ----------
    df: pandas.DataFrame
        Input data.
    col_num: str
        Numerical feature which is to be plotted.
    target_cat: str
        Target categorical feature.
    figsize: (int, int)
        Figure size.
    bins: int
        Number of bins in the histogram.
    fontsize: int
        Font size of xlabel and ylabels of all plots.
    ofile: str
        Name of output plot. eg. images/mycol_target.png
    save: bool
        Whether to save the image or not.
    disp: bool
        Display output dataframe or not.
    print_: bool
        Print output dataframe or not.
    show: bool
        Wheter or not to show the image.
    Examples
    ---------
    .. code-block:: python

        col_num = 'age'
        target_cat = 'conversion'
    """

    df = df.dropna(subset=[col_num])
    a,b = sorted(df[target_cat].dropna().unique())
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
    ax[0][0].set_xlabel(target_cat,fontsize=fontsize)
    ax[0][0].set_ylabel(col_num,fontsize=fontsize)
    ax[0][1].set_xlabel(target_cat,fontsize=fontsize)
    ax[0][1].set_ylabel(col_num,fontsize=fontsize)
    ax[1][1].set_xlabel(target_cat,fontsize=fontsize)
    ax[1][1].set_ylabel(col_num,fontsize=fontsize)

    ax[1][0].set_xlabel(col_num,fontsize=fontsize)
    ax[1][0].set_ylabel(target_cat,fontsize=fontsize)

    ax[1][0].legend()
    plt.suptitle(f"Plot of {col_num} vs {target_cat}")
    plt.tight_layout()

    # save
    if not ofile:
        ofile = f'images/{col_num}_{target_cat}.png'

    if save:
        if not os.path.isdir('images'): os.makedirs('images')
        plt.savefig(ofile)
    if show:
        plt.show()

    # display
    if disp:
        display(df.groupby(target_cat)[col_num]
        .describe().round(2).T.add_prefix(f'{target_cat}_').T
        .append(df[col_num].describe().round(2)))

    if print_:
        print(df.groupby(target_cat)[col_num]
        .describe().round(2).T.add_prefix(f'{target_cat}_').T
        .append(df[col_num].describe().round(2)))


def plot_cat_num(df,cat,num,
                    comma=False,decimals=2,rot=30,add_text=True,fontsize=14,
                    ofile=None,save=False,show=False):
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


def plot_cat_cat(self,cat,binn,figsize=(12,12),ylim2=None,rot=30,comma=True,
                loc='lower left',
                ofile=None,save=False,show=False):
    """Plot 2*2 plot for categorical feature vs target-categorical feature.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cat: str
        categorical feature.
    binn: str
        Binary feature.
    figsize: (int, int)
        Figure size.
    ylim2: int
        Second plot y-limit upper range.
    rot: int
        Degree to rotate the text on barplot.
    comma: bool
        Whether or not to style the number with comma in text in barplots.
    loc: str
        matplotlib plot loc (location) variable. 0 1 2 3 or 'upper right' etc.
    ofile: str
        Name of output plot. eg. images/mycol_target.png
    save: bool
        Whether to save the image or not.
    show: bool
        Wheter or not to show the image.

    Examples
    ---------
    .. code-block:: python

        cat = 'sex'
        target_cat = 'conversion'
    """

    # data
    rare = df[binn].value_counts().idxmin()
    df1 = df.groupby([cat,binn]).count().iloc[:,0].unstack().sort_values(rare)
    order = df1.index.tolist()

    # top right
    df1_pct = df1.sum(axis=1).div(df1.sum().sum()).mul(100).round(2)

    # mid right
    df1_pct_cat =  df1.div(df1.sum()).mul(100).round(2)
    df1_pct_cat_sorted = df1_pct_cat.sort_values(rare,ascending=False)
    df1_pct_cat_order = df1_pct_cat.loc[order]

    # bottom right
    df1_per_cat = df1.T.div(df1.sum(axis=1)).mul(100).T.round(2)
    df1_per_cat_order = df1_per_cat.loc[order]
    df1_per_cat_sorted = df1_per_cat.sort_values(rare,ascending=False)

    pal = np.random.choice(['magma','Paired', 'inferno',
                            'Spectral', 'RdBu', 'BrBG','PRGn',
                            'seismic','bwr','PuOr','twilight'])

    fig, ax = plt.subplots(3,2,figsize=figsize)

    # top left count plot
    df1.sum(axis=1).rename(binn).plot.bar(
        color=sns.color_palette(pal,df[cat].nunique()),ax=ax[0][0])
    add_text_barplot(ax[0][0],rot=rot,comma=comma)
    ax[0][0].legend(loc=loc)

    # top right percent plot
    df1_pct.rename(binn).plot.bar(color=sns.color_palette(pal,df[cat].nunique()),
                    ax=ax[0][1])
    add_text_barplot(ax[0][1], percent=True,rot=rot)
    ax[0][1].legend(loc=loc)

    # mid left count plot
    df1.plot.bar(
    color=sns.color_palette(pal,df[cat].nunique()),ax=ax[1][0])
    add_text_barplot(ax[1][0],rot=rot,comma=comma)
    ax[1][0].legend(loc='best')

    # mid right percent plot
    df1_pct_cat_order.plot.bar(ax=ax[1][1])
    add_text_barplot(ax[1][1], percent=True,rot=rot)
    ax[1][1].legend(loc='best')

    # bottom left count plot
    df1.plot.bar(
    color=sns.color_palette(pal,df[cat].nunique()),ax=ax[2][0])
    add_text_barplot(ax[2][0],rot=rot,comma=comma)
    ax[2][0].legend(loc='best')

    # bottom right
    df1_per_cat_order.plot.bar(ax=ax[2][1])
    add_text_barplot(ax[2][1], percent=True,rot=rot)
    ax[2][1].legend(loc='best')

    # limits
    if ylim2:
        ax[1][1].set_ylim(0,ylim2)

    # title
    plt.suptitle(f'Count and Percent plot for {cat} vs {binn}',
            fontsize=14,color='blue')

    # decorate
    plt.tight_layout()

    # save
    if not ofile:
        ofile = f'images/{cat}_{binn}.png'

    if save:
        if not os.path.isdir('images'): os.makedirs('images')
        plt.savefig(ofile)

    if show:
        plt.show()

    # print
    print('='*50)
    print(f'Feature: **{cat}**')
    print('Overall Count: ')
    for i,v in df[cat].value_counts(ascending=False,normalize=True
            ).mul(100).round(2).items():
        print(f'    {i}: {v}%')

    print()
    print(f'Total  **{binn}_{rare}** distribution:')
    for i,v in df1_pct_cat_sorted[rare].items():
        print(f'    {i}: {v}%')

    print()
    print(f'Per {cat}  **{binn}_{rare}** distribution:')
    for i,v in df1_per_cat_sorted[rare].items():
        print(f'    {i}: {v}%')

#=================================================
def plot_date_cat(self,col_date,target_cat,figsize=(8,6),
                    show=True,save=False):
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
def plot_daily_cat(self,col_date,target_cat,figsize=(12,8),save=False,show=False,show_xticks=True):
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

#=====================================================
    def plot_boxplot_cats_num(self,cats,num,save=False,show=False):
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


def plot_multiple_jointplots_with_pearsonr(df,cols,target,ofile):
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
def plot_corrplot_with_pearsonr(df,cols,ofile=None):
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
def plot_count_cat(df, cat, percent=True,bottom=0,figsize=(12,8),fontsizex=18, fontsizey=18, horizontal=False,number=False,
                save=False,show=False):
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
def plot_corrplot_with_pearsonr(df,cols,save=False,show=True):
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
    g.map_lower(self._annotate_pearsonr)
    plt.tight_layout()
    if save:
        if not os.path.isdir('images'): os.mkdir('images')
        ofile = os.path.join('images','_'.join(cols) + '_corrplot_pearsonr.png')
        plt.savefig(ofile,dpi=300)
    if show:
        plt.show()
        plt.close()

#=========================================================
def plot_corr(df,cols=None,cmap='RdBu',annot=True,
                    figsize=(12,8),
                    xrot=0,yrot=90,fontsize=12,
                    save=False,show=False):
    """Correlation plot.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list,optional
        List of columns.
    cmap: str,optional
        Colormap. eg. RdBu, coolward,PuBu
    annot: bool
        Show annotation or not.
    figsize: (int,int)
        Figure size.
    xrot: int
        Rotation of xtick labels
    yrot: int
        Rotation of ytick labels
    fontsize: int
        Size of x and y ticklabels
    ofile: str
        Name of the output file.
    save: bool
        Whether or not to save the image.
    show: bool
        Whether or not to show the image.
    Examples
    ---------
    .. code-block:: python

        cols = ['sqft_living', 'sqft_living15', 'sqft_above']
        plot_corr(df,cols)

    """
    if not cols:
        cols = df.columns

    plt.figure(figsize=figsize)

    g = sns.heatmap(df[cols].corr(),
                    vmin=-1,
                    vmax=1,
                    cmap=cmap,
                    annot=annot)

    g.set_yticklabels(g.get_yticklabels(),
                    rotation=yrot, fontsize=fontsize)
    g.set_xticklabels(g.get_xticklabels(),
                    rotation=xrot, fontsize=fontsize)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    if save:
        if not os.path.isdir('images'): os.mkdir('images')
        ofile = os.path.join('images',f'corrplot_' + '_'.join(cols) + '.png')
        plt.savefig(ofile,dpi=300)
    if show:
        plt.show()
        plt.close()

#==========================================================
def plot_corr_style(df,cols=None,cmap='RdBu'):
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
def plot_cat_cat2(df,cat,target_cat,
    figsize=(12,8),ylim2=None,
    ofile=None,save=False,show=False):
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
def plot_num_cat2(df, col_num,target_cat,
                                    figsize=(12,8),bins=100,
                                    odir='images'):
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

def plot_pareto(df_,num,cat,thr=None,
                figsize=(12,8),
                rot=90,fontsize=18,
                offset=0, decimals=2,
                save=False,show=False):
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
    df = df_[[cat,num]].sort_values(num,ascending=False)
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
def plot_cat_cat_pct(df, col_cat,col_cat2,
                                    figsize=(12,8),
                                    odir=None):
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
    if odir: plt.savefig(f'{odir}/{col_cat}_vs_{col_bin}.png')
    plt.show()

#=============== Plot Helper Functions ==========================
def _annotate_pearsonr(x, y, **kws):
    from scipy import stats
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("pearsonr = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)