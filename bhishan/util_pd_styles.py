from __future__ import annotations

__doc__ = "Pandas utility"
__note__ = """
Reference: https://github.com/fastai/fastai2/blob/master/fastai2/tabular/core.py

"""
__author__ = "Bhishan Poudel"

__all__ = [
    "highlight_row",
    "highlight_col",
    "highlight_diag",
    "highlight_rowf",
    "highlight_rowsf",
    "highlight_colf",
    "highlight_diagf",
    "highlight_rcd"
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
import matplotlib.pyplot as plt
import re

# local functions
try:
    from .utils import ifnone
except:
    from utils import ifnone

"""
Built-in functions:

df.style.highlight_max(color='darkorange', axis=None) # axis=None is max of whole dataframe
df.style.highlight_min(subset=['B'], axis=0) # axis=0 columnwise (default) and axis=1 for row wise
df.style.highlight_null('salmon') # null_color='red' is too bad
df.style.background_gradient(cmap='viridis',low=.5, high=0)
df.style.highlight_max(subset= pd.IndexSlice[1:3, ['B', 'D']]) # only max from given index range
df.style.bar(subset=['A', 'B'], align='mid', color=['#d65f5f', '#5fba7d'])
df.style.hide_columns(['C','D'])
df.style.hide_index()
df.style.set_caption('caption.')

"""

def highlight_row(
    ser: Series,
    color: str ="lightblue",
    row: SIN=None
    ) -> List:
    if row is None:
        row = ser.index[-1]
    bkg = f"background-color: {color}"
    return [bkg if ser.name == row else "" for _ in ser]

def highlight_col(
    ser: Series,
    color: str ="salmon",
    col: SIN=None
    ) -> List:
    if col is None:
        col = ser.index[-1]
    bkg = f"background-color: {color}"
    return [bkg if ser.name == col else "" for _ in ser]

def highlight_diag(
    dfx: DataFrame,
    color: str ="khaki"
    ) -> DataFrame:
    a = np.full(dfx.shape, "", dtype="<U24")
    np.fill_diagonal(a, f"background-color: {color}")
    df1 = pd.DataFrame(a, index=dfx.index, columns=dfx.columns)
    return df1

def highlight_rowf(
    dfx : Dataframe,
    color :str ="lightblue",
    row: SIN =None
    ) -> Any:
    if row is None:
        row = dfx.index[-1]

    def highlight_row(
        ser: Series,
        row: SIN,
        color: str="salmon"
        ) -> List:
        bkg = f"background-color: {color}"
        return [bkg if ser.name == row else "" for _ in ser]

    return dfx.style.apply(highlight_row, axis=1, row=row)

def highlight_rowsf(
    dfx: DataFrame,
    color: str ="lightblue",
    rows: SIN =None
    ) -> Styler:
    if rows is None:
        rows = [dfx.index[-1]]

    def highlight_row(ser: Series,
        row: SI,
        color: str ="salmon"
        ) -> List:
        bkg = f"background-color: {color}"
        return [bkg if ser.name == row else "" for _ in ser]

    df_style = dfx.copy()
    df_style = df_style.style.apply(highlight_row, axis=1, row=rows[0])

    for row in rows[1:]:
        df_style = df_style.apply(highlight_row, axis=1, row=row)

    return df_style

def highlight_colf(
    dfx: DataFrame,
    color: str ="salmon",
    col: SI =-1
    ) -> Styler:
    if not isinstance(col, str):
        col = dfx.columns[col]

    def highlight_col(dfy: DataFrame) -> DataFrame:  # axis=None needs dataframe
        bkg = f"background-color: {color}"
        df1 = pd.DataFrame("", index=dfy.index, columns=dfy.columns)
        df1.loc[:, col] = bkg
        return df1

    return dfx.style.apply(highlight_col, axis=None)

def highlight_diagf(
    dfx: DataFrame,
    color: str ="khaki"
    )-> Styler:
    def highlight_diag(dfy:DataFrame)->DataFrame:
        a = np.full(dfy.shape, "", dtype="<U24")
        np.fill_diagonal(a, f"background-color: {color}")
        df1 = pd.DataFrame(a, index=dfy.index, columns=dfy.columns)
        return df1

    return dfx.style.apply(highlight_diag, axis=None)

def highlight_rcd(
    dfx: DataFrame,
    row: SIN =None,
    col: SIN =None,
    c1: str ="lightblue",
    c2: str ="salmon",
    c3: str ="khaki"
    )-> Styler:
    return (
        dfx.style.apply(highlight_diag, axis=None, color=c3)
        .apply(highlight_row, axis=1, color=c1, row=row)
        .apply(highlight_col, axis=0, color=c2, col=col)
    )

def style_rowcol(
    df:DataFrame,
    name:Any=0,
    axis:Union[str,int]=1,
    color:SN=None,
    c:SN=None,
    )-> Styler:
    """Style rows and columns of pandas dataframe.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    name: int or str, or tuple
        Index name of row. e.g 0, 'myindex', (level0, level1)
    axis: int or str
        axis =1 for highlight row
    color: str
        Color of the row
    c: str
        Alias for color

    Examples
    ---------
    .. code-block:: python
        df = sns.load_dataset('titanic')
        df1 = df.head()
        df2 = df.groupby(['sex', 'class']).agg({'fare': ['sum','count']})

        df1.bp.style_rowcol(2)
        df1.bp.style_rowcol([0,2],'khaki')

        df2.bp.style_rowcol(('male','First'))
        df2.bp.style_rowcol(['male','First'])
        df2.bp.style_rowcol(('*','First'))
        df2.bp.style_rowcol([('male','First'),('female','Second')])

        df2.bp.style_rowcol([('fare','sum')],axis=0)
    """
    row_color = 'lightblue'
    if axis == 'row':
        axis = 1
    if str(axis).startswith('co'):
        axis = 0
    if color:
        row_color = color
    if c:
        row_color = c

    # make list of index names
    if type(name) in [str,int]:
        names = [name]
    else:
        names = name

    multi_index = False
    if type(df.index) == pd.core.indexes.multi.MultiIndex:
        multi_index = True

    if multi_index:
        cond = (any(isinstance(el, list) for el in name) or
                any(isinstance(el, tuple) for el in name))
        names = [name] if not cond else name
        names = [list(i) for i in names]

        if names[0][0] == '*':
            return df.style.apply(lambda ser: [f'background: {row_color}'
                        if ser.name[1] == names[0][1]
                        else ''
                        for _ in ser],axis=axis)
        if names[0][1] == '*':
            return df.style.apply(lambda ser: [f'background: {row_color}'
                        if ser.name[0] == namees[0][0]
                        else ''
                        for _ in ser],axis=axis)
        else:
            return df.style.apply(lambda ser: [f'background: {row_color}'
                if  list(ser.name) in names
                else ''
                for _ in ser],axis=axis)
    # not multiindex
    if not multi_index:
        return df.style.apply(lambda ser: [f'background: {row_color}'
                if ser.name in names
                else ''
                for _ in ser],axis=axis)

def style_diag(
    df:DataFrame,
    diag:str='both',
    c1:str='lightgreen',
    c2:str='salmon'
    )-> Styler:
    """Style rows and columns of pandas dataframe.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    diag: int or str
        Highlight which diagonal? 'first', 'second', 'both', 0,1,2
    c1,c2: str
        Color for diagonals

    Examples
    ---------
    .. code-block:: python
        df = pd.DataFrame(data={'p0': [10,   4],'pred1': [0,   0],
                'total': [10,  4]},index=['true0','true1'] )
        df.bp.style_diag(diag=0)

    """
    def highlight_diags(dfx,c1='lightgreen',c2='salmon',diag='both'):
        attr1 = f'background-color: {c1}'
        attr2 = f'background-color: {c2}'

        df_style = dfx.replace(dfx, '')
        if diag==0 or diag == 'first':
            np.fill_diagonal(df_style.values, attr1)
        if diag==1 or diag == 'second':
            np.fill_diagonal(np.flipud(df_style), attr2)
        if diag==2 or diag=='both':
            np.fill_diagonal(df_style.values, attr1)
            np.fill_diagonal(np.flipud(df_style), attr2)
        return df_style

    return df.style.apply(highlight_diags,diag=diag,c1=c1,c2=c2,axis=None)

def style_cellv(
    df:DataFrame,
    cond:str='',
    c:str='lightgreen',
    idx:SIN=None,
    col:SIN=None
    )-> Styler:
    """Style rows and columns of pandas dataframe.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cond: str
        Mask. eg. "x[x.index==2]==0"
    c: str
        Color of style.
    idx: str
        Index name.
    col: str
        Column name.

    Examples
    ---------
    .. code-block:: python
        df = sns.load_dataset('titanic')
        df1 = df.head()
        cond = "type(v) in [int,float] and 1<=v<=3"
        cond = "v in [3,'S']"
        df1.bp.style_cellv(cond)
        df1.bp.style_cellv("v==1",idx=2)
        df1.bp.style_cellv("v==1",col='pclass')

    """
    if idx !=None and col !=None:
        return df.style.apply(lambda x: [f"background: {c}"
                if (df.columns[i] == col and x.name == idx)
                else ""
                for i, v in enumerate(x)], axis = 1)

    if cond.replace(" ",'') in ["v==0","v==1"]:
        cond = "type(v) in [int,float] and " + cond
    if idx !=None:
        return df.style.apply(lambda x: [f"background: {c}"
                if x.name == idx and eval(cond)
                else ""
                for v in x], axis=1)

    if col !=None:
        return df.style.apply(lambda x: [f"background: {c}"
                if x.name == col and eval(cond)
                else ""
                for v in x], axis=0)

    return df.style.apply(lambda x: [f"background: {c}"
                if eval(cond) else "" for v in x], axis=1)


def style_cellx(
    df:DataFrame,
    cond:str,
    c:str='lightgreen'
    )-> Styler:
    """Style rows and columns of pandas dataframe.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cond: str
        Mask. eg. "x[x.index==2]==0"
    c: str
        Color of style.

    Examples
    ---------
    .. code-block:: python
        df = sns.load_dataset('titanic')
        df1 = df.head()
        cond = "x[x.index==2]==0"
        cond = "x==1"
        cond = "x.age==35"
        df1.bp.style_cellx(cond)
        df1.bp.style_cellx("x['survived']==x['pclass']")

    """
    def f(x,cond,c='lightgreen'):
        attr = f'background-color: {c}'
        df1 = x.astype(str).replace(x, '')
        df1 = df1.astype(str).replace('nan','')
        df1[eval(cond)] = attr
        return df1
    return df.style.apply(f,axis=None,cond=cond,c=c)




