__author__ = 'Bhishan Poudel'

__doc__ = """
This module helps fitting various data science tools.

- print_df_eval()
- freq_count(df,nlargest=None)
- get_column_descriptions(df, column_list=None,style=True)
- adjustedR2(r2,nrows,kcols)
- multiple_linear_regression(df,features,target,model,cv=5)
- get_most_correlated_features(df,print_=False,thresh=0.5)

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    help(bp.ds_ds)

"""

import numpy as np
import pandas as pd
import sklearn

def print_df_eval():
    """Print a data frame to store the model evalution.
    """
    ans = """df_eval = pd.DataFrame({'Model': [],
                        'Details':[],
                        'Root Mean Squared Error (RMSE)':[],
                        'R-squared (training)':[],
                        'Adjusted R-squared (training)':[],
                        'R-squared (test)':[],
                        'Adjusted R-squared (test)':[],
                        '5-Fold Cross Validation':[]})"""
    print(ans)

def freq_count(df,nlargest=None):
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
    vals = df.values.T
    freq_counter = [ collections.Counter(vals[i]).most_common(nlargest)
            for i in range(len(vals)) ]
    df_freq = pd.DataFrame(np.array(freq_counter).T,
                            columns=['item_frequency_tuple'],
                            index=df.columns)
    return df_freq

def get_column_descriptions(df, column_list=None,style=True):
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

    return(df_desc)

def adjustedR2(r2,nrows,kcols):
    """Function to calculate adjusted R-squared.

    R-squared metric increases with number of features used.
    To get more robust model comparison metric, we can use
    adjusted R-squared metric which penalizes larger number of featrues.

    (R-adj)^2 = R^2 - (k-1)/(n-k) * (1-R^2)

    Parameters
    ----------
    r2: float
        The r-squared value.
    nrows: int
        Total number of observations.
    kcols: int
        Total number of parameters.

    """
    return r2-(kcols-1)/(nrows-kcols)*(1-r2)

def multiple_linear_regression(df,features,target,model,cv=5):
    """ Multiple Linear Regression Modelling using given model.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe.
    features: list
        list of feature names
    target: string
        target name.
    model: model
        sklearn model e.g. sklearn.ensemble.RandomForestRegressor
    cv: int
        cross-validation

    """
    # train test values
    X = df[features].values
    y = df[target].values.reshape(-1,1)

    Xtrain = train[features].values
    ytrain = train[target].values.reshape(-1,1)

    Xtest = test[features].values
    ytest = test[target].values.reshape(-1,1)

    # random forest regressor wants 0d scalar not 1d vector
    if isinstance(model, sklearn.ensemble.RandomForestRegressor):
        ytrain = train[target].values.ravel()

    # fitting
    model.fit(Xtrain,ytrain)

    # prediction
    ypreds = model.predict(Xtest)

    # metrics
    rmse = np.sqrt(mean_squared_error(ytest,ypreds)).round(3)
    r2_train = model.score(Xtrain, ytrain).round(3)
    r2_test = model.score(Xtest, ytest).round(3)

    cv = cross_val_score(model, X, y, cv=5,n_jobs=-1,verbose=1).mean().round(3)

    ar2_train = adjustedR2(model.score(Xtrain,ytrain),
                            Xtrain.shape[0],
                            len(features)).round(3)
    ar2_test  = adjustedR2(model.score(Xtest,ytest),
                            Xtest.shape[0] ,
                            len(features)).round(3)

    return (rmse, r2_train, ar2_train, r2_test, ar2_test, cv)

def get_high_correlated_features_df(df,print_=False,thr=0.5):
    """Get the most correlated features above given threshold.

    Note:
    1. Only numerical features have correlation.
    2. Here we only get absolute correlation.

    """

    df1 = (df.corr()
    .abs()
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
    .query('corr > @thr')
    )
    if print_:
        print(df1)
    return df1

    return df1