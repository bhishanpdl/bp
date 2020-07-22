__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various data preprocessing utilities.

- outliers_tukey(x,thresh=1.5)
- outliers_kde(x)
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

__all__ = ["outliers_tukey",
        "outliers_kde",
        "add_interactions",
        "select_kbest_features"
        ]

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype

def outliers_tukey(ser,thr=1.5,plot_=False,show=False):
    """Get outliers bases on Tukeys Inter Quartile Range.

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

    Example
    ---------
    .. code-block:: python

        ser_outliers = outliers_tukey(df['age'])
    """
    ser = ser.dropna()
    q1 = np.percentile(ser, 25)
    q3 = np.percentile(ser, 75)
    iqr = q3-q1
    floor = q1 - thr*iqr
    ceiling = q3 + thr*iqr
    idx_outliers = list(ser.index[(ser < floor)|(ser > ceiling)])
    ser_outliers = ser.loc[idx_outliers].to_frame()

    if plot_:
        sns.boxplot(x=ser)
    if show:
        plt.show()

    return ser_outliers

def outliers_kde(x):
    """Find outliers using KDEUnivariate method.

    Parameters
    -----------
    x: np.array
        1d array or list or series whose outliers are to be found.
        The array must NOT have nans.

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

def remove_outlier(df_in, col):
    """Remove outliers using IQR method.

    Parameters
    -----------
    df_in: pandas.DataFrame
        Input pandas dataframe.
    col: str
        Name of column

    Examples
    ---------
    .. code-block:: python

            df = sns.load_dataset('titanic')
            df_no_age_outliers = bp.remove_outliers(df,'age')

    """
    if not is_numeric_dtype(df[col]):
        raise AttributeError(f'{col} must be a numeric column')

    q1 = df_in[col].quantile(0.25)
    q3 = df_in[col].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    low  = q1-1.5*iqr
    high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col] > low) & (df_in[col] < high)]
    return df_out

def add_interactions(df):
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

def select_kbest_features(df_Xtrain, df_ytrain, df_Xtest):
    """Select Kbest features using scikit learn SelectKBest.

    Parameters
    -----------
    df_Xtrain: pandas.DataFrame
        Training predictor variables.
    df_ytest: pandas.DataFrame
        Training response variable.
    df_Xtest: pandas.DataFrame
        Test predictor variables.

    Returns
    --------
    df_Xtrain_selected: pandas.DataFrame
        Training data with selected features.
    df_Xtest_selected: pandas.DataFrame
        Test data with selected features.
    """
    import sklearn.feature_selection
    select = sklearn.feature_selection.SelectKBest(k=20)
    selected_features = select.fit(df_Xtrain,  df_ytrain)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [df_Xtrain.columns[i] for i in indices_selected]

    df_Xtrain_selected = df_Xtrain[colnames_selected]
    df_Xtest_selected = df_Xtest[colnames_selected]
    return df_Xtrain_selected, df_Xtest_selected