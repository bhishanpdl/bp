__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various data preprocessing utilities.

- outliers_tukey(x,thresh=1.5)
- outliers_kde(x)
- remove_outliers_iqr
- add_interactions(df)
- select_kbest_features(df_Xtrain, df_ytrain, df_Xtest)

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.ml_data_preprocessing?

"""

__all__ = [
    "get_outliers",
    "get_outliers_tukey",
    "get_outliers_kde",
    "remove_outliers",
    "remove_outliers_iqr",
    "remove_outliers_tukey",
    "add_interactions",
    "select_kbest_features"
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
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype

def get_outliers(
    ser:Series,
    k: NUM=1.5,
    plot_: bool=False,
    show: bool=False,
    info: bool=True
    ):
    """Get outliers based on John Tukey's Inter Quartile Range IQR method.

    We remove following outliers:
        values < Q1 - 1.5*IQR
        values > Q3 + 1.5*IQR

    where, IQR = Q3 - Q1.

    According to Tukey, k=1.5 gives outlier and k=3 gives far out points.

    Parameters
    -----------
    ser: pandas series
        Data whose outliers is to be found.
    thr: float
        Thereshold.
    plot_: bool
        Whether or not to plot.
    show: bool
        Whether or not to show the plot.
    info: bool
        Whether or not print how many points are removed.

    Returns
    --------
    ser_outliers: pd.Series
        Pandas series within 25 <= values <= 75 quantiles.

    Example
    ---------
    .. code-block:: python

        ser_outliers = outliers_tukey(df['age'])

    References
    -----------
    https://en.wikipedia.org/wiki/Outlier

    """
    ser = ser.dropna()
    q1 = np.percentile(ser, 25)
    q3 = np.percentile(ser, 75)

    iqr = q3-q1
    floor   = q1 - k*iqr
    ceiling = q3 + k*iqr
    idx_outliers = list(ser.index[(ser < floor)|(ser > ceiling)])
    ser_outliers = ser.loc[idx_outliers].to_frame()

    n_removed = len(ser_outliers)
    n_pct = n_removed / len(ser) * 100
    if info:
        print("Here we get outliers index and values.")
        print("We may need to removed these outliers later.")
        print(f"number of rows removed = {n_removed} ({n_pct}%)")

    if plot_:
        sns.boxplot(x=ser)
    if show:
        plt.show()

    return ser_outliers

# alias
get_outliers_iqr = get_outliers
get_outliers_tukey = get_outliers

def get_outliers_kde(x:ARR):
    """Find outliers using KDEUnivariate method.

    Parameters
    -----------
    x: np.array
        1d array or list or series whose outliers are to be found.
        The array must NOT have NaNs.

    Returns
    --------
    idx_outliers: np.array
        Index of outliers based on KDEUnivariate Method.
    val_outliers: np.array
        Values of outliers based on KDEUnivariate Method.

    NOTE
    ----
    This method uses nonparametric way to estimate outliers.
    It captures the outliers even in cases of bimodal distributions.

    Ref: http://www.statsmodels.org/devel/generated/statsmodels.nonparametric.kde.KDEUnivariate.html

    Examples
    ---------
    .. code-block:: python

        kde_indices, kde_values = bp.outliers_kde(df['age'].dropna())
        print(np.sort(kde_values))

    """
    from sklearn.preprocessing import scale
    from statsmodels.nonparametric.kde import KDEUnivariate

    assert np.isnan(x).sum() == 0, 'Missing values are not allowed'

    x_scaled = scale(list(map(float, x)))
    kde = KDEUnivariate(x_scaled)
    kde.fit(bw="scott", fft=True)
    pred = kde.evaluate(x_scaled)

    n = sum(pred < 0.05)
    idx_outliers = np.asarray(pred).argsort()[:n]
    val_outliers = np.asarray(x)[idx_outliers]

    return idx_outliers, val_outliers

def remove_outliers(
    df:DataFrame,
    col:SI,
    k: NUM=1.5,
    info: bool=True
    ):
    """Remove outliers using John Tukey's IQR method.

    We remove following outliers:
        values < Q1 - 1.5*IQR
        values > Q3 + 1.5*IQR

    where, IQR = Q3 - Q1.

    According to Tukey, k=1.5 gives outlier and k=3 gives far out points.

    Parameters
    -----------
    df: pandas.DataFrame
        Input pandas dataframe.
    col: str
        Name of column
    info: bool
        Whether or not print additional information.

    Returns
    --------
    df_out: pd.DataFrame
        Output dataframe with given column outliers removed based on IQR.

    Examples
    ---------
    .. code-block:: python

            df = sns.load_dataset('titanic')
            df_no_age_outliers = bp.remove_outliers_iqr(df,'age')

    """
    if not is_numeric_dtype(df[col]):
        raise AttributeError(f'{col} must be a numeric column')

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3-q1 #Interquartile range
    low  = q1 - k*iqr
    high = q3 + k*iqr
    df_out = df.loc[(df[col] > low) & (df[col] < high)]
    n_removed = df.shape[0] - df_out.shape[0]
    n_pct = n_removed / df.shape[0] * 100

    if info:
        print(f"For {col}, number of rows removed = {n_removed} ({n_pct}%)")
    return df_out

# alias
remove_outliers_iqr = get_outliers
remove_outliers_tukey = get_outliers

def add_interactions(df:DataFrame):
    """Add two way interaction feature terms for all columns of pandas dataframe.

    Parameters
    -----------
    df: pandas.DataFrame
        Data to which we want to add the features.

    Examples
    ---------
    .. code-block:: python

            df = add_interactions(df)

    NOTE
    -----
    This gives a lot of features which might cause the model to overfit.
    We can select k best features using sklearn after doing this:

    .. code-block:: python

        import sklearn.feature_selection
        select = sklearn.feature_selection.SelectKBest(k=20)
        selected_features = select.fit(df_Xtrain,  df_ytrain)
        indices_selected = selected_features.get_support(indices=True)
        colnames_selected = [df_Xtrain.columns[i] for i in indices_selected]

        df_Xtrain_selected = df_Xtrain[colnames_selected]
        df_ Xtest_selected = df_Xtest[colnames_selected]

    """

    # Get feature names
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]

    # Find interactions
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames

    # Remove interaction terms with all 0 values
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis=1)

    return df

def select_kbest_features(
    df_Xtrain:DataFrame,
    df_ytrain:ARR,
    df_Xtest:DataFrame,
    k:int=20
    ):
    """Select Kbest features using scikit learn SelectKBest.

    Parameters
    -----------
    df_Xtrain: pandas.DataFrame
        Training predictor variables.
    df_ytest: array-like
        Training response variable.
    df_Xtest: pandas.DataFrame
        Test predictor variables.
    k: int
        Number of features to select.

    Returns
    --------
    df_Xtrain_selected: pandas.DataFrame
        Training data with selected features.
    df_Xtest_selected: pandas.DataFrame
        Test data with selected features.
    """
    import sklearn.feature_selection

    select = sklearn.feature_selection.SelectKBest(k=k)
    selected_features = select.fit(df_Xtrain,  df_ytrain)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [df_Xtrain.columns[i] for i in indices_selected]

    df_Xtrain_selected = df_Xtrain[colnames_selected]
    df_Xtest_selected = df_Xtest[colnames_selected]
    return df_Xtrain_selected, df_Xtest_selected