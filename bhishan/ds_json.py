__author__ = 'Bhishan Poudel'

__doc__ = """
This module provides various tools for working with json files.

- parse_json_col(df,json_col)

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    help(bp.ds_json)

"""
__all__ = ['parse_json_col','MyJson']

# Imports
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
from pandas.core.frame import DataFrame, Series
from mytyping import (IN, SN, SI, SIN, TL, LD, DS, DSt, NUM, NUMN,
                        AD, AS, DN, ARR, ARRN, SARR, LIMIT, LIMITN)

import json

def parse_json_col(df:DataFrame,json_col:SI)->DataFrame:
    """Explode the json column and attach to original dataframe.

    Parameters
    -----------
    df: pandas.DataFrame
        input dataframe
    json_col: string
        Column name of dataframe which contains json objects.

    Example
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        pd.options.display.max_colwidth=999

        df = pd.DataFrame({'id': [0],
                        'payload': [\"""{"analytics": {"device": "Desktop",
                                                        "email_open_rate_pct": 14.0},
                                        "industry": "Construction",
                                        "time_in_product_mins": 62.45}\"""]
                        })

    """
    # give increasing index to comine later
    df = df.reset_index().copy()

    df_json = df[json_col].apply(json.loads).apply(pd.io.json.json_normalize)
    df_json = pd.concat(df_json.to_numpy())
    df_json.index = range(len(df_json))

    df_no_json = df.drop(json_col,axis=1)
    cols = df_no_json.columns.tolist() + df_json.columns.tolist()

    df_combined = pd.concat([df_no_json, df_json], axis=1, ignore_index=False)
    df_combined.columns = cols

    # retrieve the original index
    df_combined.set_index('index',inplace=True)
    return df_combined

class MyJson(object):
    def __init__(self:Any):
        pass

    def parse_json_col(self:Any, df:DataFrame,json_col:str)->DataFrame:
        """Explode the json column and attach to original dataframe.
        Parameters
        -----------
        df: pandas.DataFrame
            input dataframe
        json_col: string
            Column name of dataframe which contains json objects.
        Example:
        --------
        import numpy as np
        import pandas as pd
        pd.options.display.max_colwidth=999
        from bp.ds_json import MyJson
        df = pd.DataFrame({'id': [0],
                        'payload': [\"""{"analytics": {"device": "Desktop",
                                                        "email_open_rate_pct": 14.0},
                                        "industry": "Construction",
                                        "time_in_product_mins": 62.45}\"""]
                        })
        mj = MyJson()
        ans = mj.parse_json_col(df,'payload')
        """
        # give increasing index to combine later
        df = df.reset_index()

        df_json = df[json_col].astype(str).apply(json.loads).apply(pd.json_normalize)
        df_json = pd.concat(df_json.to_numpy())
        df_json.index = range(len(df_json))

        df_no_json = df.drop(json_col,axis=1)
        cols = df_no_json.columns.tolist() + df_json.columns.tolist()

        df_combined = pd.concat([df_no_json, df_json], axis=1, ignore_index=False)
        df_combined.columns = cols

        # retrieve the original index
        df_combined.set_index('index',inplace=True)
        return df_combined