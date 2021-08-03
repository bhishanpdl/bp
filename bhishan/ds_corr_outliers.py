__author__ = 'Bhishan Poudel'

__doc__ = """
This module helps fitting various data science tools.

- print_df_eval()
- freq_count(df,nlargest=None)
- get_column_descriptions(df, column_list=None,style=True)
- adjustedR2(r2,nrows,kcols)
- multiple_linear_regression(df,features,target,model,cv=5)
- get_high_correlated_features_df(df,print_=False,thresh=0.5)

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    help(bp.ds_ds)

"""

__all__ = [
    'corrwith',
    'corr_high',
    'corr_high_lst',
    'partial_corr',
    'point_biserial_correlation',
    'outliers_tukey',
    'outliers_kde'
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

# imports
import numpy as np
import pandas as pd
import sklearn
import os
from functools import reduce,wraps
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

# local functions
try:
    from .util_plots import (add_text_barplot, magnify,
                        get_mpl_style, get_plotly_colorscale)
except:
    from util_plots import (add_text_barplot, magnify,
                        get_mpl_style, get_plotly_colorscale)

def corrwith(
    df:DataFrame,
    target:SI,
    cols:ARR=None,
    method:str='spearman'
    )->DataFrame:
    """Correlation between multiple columns with target column.

    Available methods:
    method : {'pearson', 'kendall', 'spearman'}
    """
    if cols is None:
        cols = [i for i in df.columns if i != target]
    df_corr = (df[cols].corrwith(df[target],method=method)
                    .rename(method+'r')
                    .rename_axis('column')
                    .reset_index())
    return df_corr

def corr_high(
    df:DataFrame,
    thr:NUM,
    print_:bool=False,
    disp:bool=True
    ):
    """Get the most correlated features above given threshold.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe.
    thr: float
        Thereshold between 0 and 1.
    print_: bool
        Whether or not to print the output dataframe.

    Note:
    1. Only numerical features have correlation.
    2. Here we only get absolute correlation.

    Returns
    ----------
    df1 : pandas.DataFrame
        Output dataframe with most correlated features.

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        df.corr().style.apply(lambda x: ["background: salmon" if  (abs(v) > 0.5 and v!=1) else "" for v in x], axis = 1)
        df.bp.corr_high(thr=0.5)

    """
    # df = self._obj
    df_high_corr = (df.corr()
    .unstack()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={'level_0':'feature1',
                        'level_1':'feature2',
                        0:'corr'})
    .query('feature1 != feature2')
    .assign(
    tmp = lambda dfx: dfx[['feature1', 'feature2']]\
                .apply(lambda x: '_'.join(sorted(tuple(x))),
                    axis=1)
        )
    .drop_duplicates('tmp')
    .drop('tmp',axis=1)
    .query(' abs(corr) > @thr')
    .reset_index(drop=True)
    )

    # print useful info
    cols_high_corr = list(set(df_high_corr[['feature1',
                        'feature2']].to_numpy().ravel()))
    cols_high_corr1 = df_high_corr['feature1'].to_list()
    cols_high_corr2 = df_high_corr['feature2'].to_list()
    cols_high_corr_drop = np.setdiff1d(cols_high_corr1,cols_high_corr2).tolist()

    print(f'cols_high_corr = {cols_high_corr}')
    print(f'cols_high_corr1 = {cols_high_corr1}')
    print(f'cols_high_corr2 = {cols_high_corr2}')
    print(f'cols_high_corr_drop = {cols_high_corr_drop}')

    # print
    if print_:
        print(df_high_corr)

    # display
    if disp:
        return display(df_high_corr.style.background_gradient(subset=['corr']))

    return df_high_corr

def corr_high_lst(
    df:DataFrame,
    thr:NUM,
    print_:bool=False
    ):
    """Get the most correlated features list above given threshold.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe.
    print_: bool
        Whether or not to print the output dataframe.
    thr: float
        Thereshold between 0 and 1.

    Returns
    ----------
    cols_corr : list
        List of most correlated features.

    Note
    ------
    1. Only numerical features have correlation.
    2. Here we only get absolute correlation.

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        df.bp.corr_high_lst(thr=0.5)
    """
    # df = self._obj
    cols_high_corr_drop = set()
    df_corr = df.corr()
    for i in range(len(df_corr.columns)):
        for j in range(i):
            if (( abs(df_corr.iloc[i, j]) >= thr) and
                (df_corr.columns[j] not in cols_high_corr_drop)):
                colname = df_corr.columns[i]
                cols_high_corr_drop.add(colname)
    if print_:
        print(df1)

    cols_high_corr_drop = list(cols_high_corr_drop)
    print(f'cols_high_corr_drop = {cols_high_corr_drop}')
    return cols_high_corr_drop

#============== Partial Correlation =================
def partial_corr(
    df:DataFrame,
    cols:ARRN=None,
    print_:bool=False,
    thr:NUM=1.0,
    disp:bool=False
    ):
    """Partial correlation coefficient among multiple columns of given array.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe.
    cols: list
        list of feature names
    print_: bool
        Whether or not to print the output dataframe.
    thr: float
        Threshold between 0 and 1.
    show: bool
        Whether or not to show the styled output dataframe.

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        df.bp.partial_corr()
    """
    import scipy

    if not cols:
        cols = df.select_dtypes('number').columns

    df = df[cols].dropna(how='any').copy()
    arr = df[cols].to_numpy()
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

    if disp:
        df1 = df_partial_corr.style.apply(lambda x: ["background: salmon"
                if  (abs(v) > 0.5 and v!=1) else "" for v in x], axis = 1)
        return df1
    return df_partial_corr

def point_biserial_correlation(
    df:DataFrame,
    col1:SI,
    col2:SI
    ):
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

#============ Outliers =============
def outliers_tukey(
    df:DataFrame,
    num:SI,
    thr:NUM=1.5,
    plot_:bool=True,
    ms:SIN=None,
    odir:str='images',
    ofile:str='',
    save:bool=True,
    show:bool=False,
    dpi:int=300
    ):
    """Get outliers based on Tukey's Inter Quartile Range.

    Parameters
    -----------
    df: pandas dataframe
        Pandas dataframe.
    num: str
        Name of numerical column.
    thr: float
        Thereshold.
    plot_: bool
        Whether or not to plot.
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

    Example
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        ser_outliers = df.bp.outliers_tukey('age')
    """
    # df = self._obj
    plt.style.use(get_mpl_style(ms))
    if not is_numeric_dtype(df[num]):
        raise AttributeError(f'{num} must be a numeric column')

    # We need to drop nans
    ser = df[num].dropna()

    q1 = np.percentile(ser, 25)
    q3 = np.percentile(ser, 75)
    iqr = q3-q1
    low = q1 - thr*iqr
    high = q3 + thr*iqr
    idx_outliers = list(ser.index[(ser < low)|(ser > high)])
    ser_outliers = ser.loc[idx_outliers].to_frame()

    if plot_:
        sns.boxplot(x=ser)
        plt.title(f'Box plot of {num}')
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'{num}_boxplot.png')

        if save: plt.savefig(ofile,dpi=dpi)
        if show: plt.show(); plt.close()

    return ser_outliers

def outliers_kde(
    df:DataFrame,
    num:SI,
    thr:NUM=0.05
    ):
    """Find outliers using KDEUnivariate method.

    Parameters
    -----------
    df: pandas dataframe
        Pandas dataframe.
    num: str
        Name of numerical column. The column must NOT have nans.
    thr: float
        Thereshold. Default is 5% (ie. 0.05)
    plot_: bool
        Whether or not to plot.
    odir: str
        Name of output directory.
        This directory will be created if it does not exist.
    ofile: str
        Base name of output image.
    save: bool
        Whether or not to save the image.
    show: bool
        Whether or not to show the image.

    Example
    ---------
    .. code-block:: python
        df = sns.load_dataset('titanic')
        df1 = df.dropna(subset=['age']).reset_index(drop=True)
        idx_outliers, val_outliers = df1.bp.outliers_kde('age')
        df1.loc[idx_outliers,['age']]

    NOTE
    ----
    This method uses nonparametric way to estimate outliers.
    It captures the outliers even in cases of bimodal distributions.

    Ref: http://www.statsmodels.org/devel/generated/statsmodels.nonparametric.kde.KDEUnivariate.html

    """
    # df = self._obj
    if not is_numeric_dtype(df[num]):
        raise AttributeError(f'{num} must be a numeric column')

    assert df[num].isna().sum() == 0, 'Missing values are not allowed.'

    x = df[num].to_numpy()

    from sklearn.preprocessing import scale
    from statsmodels.nonparametric.kde import KDEUnivariate

    x_scaled = scale(list(map(float, x)))
    kde = KDEUnivariate(x_scaled)
    kde.fit(bw="scott", fft=True)
    pred = kde.evaluate(x_scaled)

    n = sum(pred < thr)
    idx_outliers = np.asarray(pred).argsort()[:n]
    val_outliers = np.asarray(x)[idx_outliers]

    return idx_outliers, val_outliers

