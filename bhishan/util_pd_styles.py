from __future__ import annotations

__doc__ = "Pandas Styler"
__note__ = """


"""
__author__ = "Bhishan Poudel"

__all__ = [
    "style_rows",
    "style_cols",
    "style_row_mi",
    "style_diags",
    "style_rowscols",
    "style_rc",
    "style_rowscolsdiags",
    "style_rcd",
    "style_cellv",
    "style_cellx",
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

info = """
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

def df_style_info():
    print(info)

#===== Helper functions =====
def _style_row(
    ser: Series,
    color: str ="lightblue",
    row: SIN=None
    ) -> List:
    if row is None:
        row = ser.index[-1]
    bkg = f"background-color: {color}"
    return [bkg if ser.name == row else "" for _ in ser]

def _style_rows(
    df:DataFrame,
    rows: Union[str,int,List,Tuple]=None,
    colors: Union[str,List,Tuple]='lightblue'
    )->DataFrame:
    # if df is Styler, make it dataframe
    if isinstance(df,pd.io.formats.style.Styler):
        df = df.data

    # if rows is None, make it first index
    if rows is None:
        rows = [df.index[0]]

    # if cols is integer or str, make it list
    if type(rows) in [int,str]:
        rows = [rows]

    # if index name is not integer, use integer to get nth index
    if not isinstance(df.index[0],int):
        if isinstance(rows[0], int):
            rows = list(df.index[rows])

    # create colors
    if colors is None:
        colors = ['salmon'] * len(rows)

    # if color is string, make list
    if isinstance(colors,str):
        colors = [colors]*len(rows)

    df_str = pd.DataFrame("", index=df.index, columns=df.columns)

    for row,color in zip(rows,colors):
        attr = f"background-color: {color}"
        if row not in df.index:
            print(f"ERROR: '{row}' is not in Index.")
        if row in df.index:
            df_str.loc[row,:] = attr
    return df_str

def _style_col(
    ser: Series,
    color: str ="salmon",
    col: SIN=None
    ) -> List:
    if col is None:
        col = ser.index[-1]
    bkg = f"background-color: {color}"
    return [bkg if ser.name == col else "" for _ in ser]

def _style_cols(
    df:DataFrame,
    cols: Union[str,int,List,Tuple]=None,
    colors: Union[str,List,Tuple]='salmon'
    )-> DataFrame:
    # if df is Styler, make it dataframe
    if isinstance(df,pd.io.formats.style.Styler):
        df = df.data

    # if cols is None, make it first column
    if cols is None:
        cols = [df.columns[0]]

    # if cols is integer or str, make it list
    if type(cols) in [int,str]:
        cols = [cols]

    # if column name is not integer, use integer to get nth column
    if not isinstance(df.columns[0],int):
        if isinstance(cols[0], int):
            cols = list(df.columns[cols])

    # create colors
    if colors is None:
        colors = ['salmon'] * len(cols)

    # if color is string, make list
    if isinstance(colors,str):
        colors = [colors]*len(cols)

    df_str = pd.DataFrame("", index=df.index, columns=df.columns)

    for col,color in zip(cols,colors):
        attr = f"background-color: {color}"
        if col not in df.columns:
            print(f"ERROR: '{col}' is not in Columns.")
        if col in df.columns:
            df_str.loc[:, col] = attr
    return df_str

def _style_diags0(
    df: DataFrame,
    color: str ="khaki"
    ) -> DataFrame:
    a = np.full(df.shape, "", dtype="<U24")
    np.fill_diagonal(a, f"background-color: {color}")
    df1 = pd.DataFrame(a, index=df.index, columns=df.columns)
    return df1

def _style_diags(
    df:DataFrame,
    diag:SI='both',
    c1:str='lightgreen',
    c2:str='salmon'
    ):
    # colors
    attr1 = f'background-color: {c1}'
    attr2 = f'background-color: {c2}'

    # empty array
    arr_str = np.full(df.shape, '', dtype='<U32')

    # main diagonal
    if diag==0 or str(diag).startswith('f'):
        np.fill_diagonal(arr_str, attr1)

    # second diagonal
    if diag==1 or str(diag).startswith('s'):
        np.fill_diagonal(np.flipud(arr_str), attr2)

    # both
    if diag==2 or diag=='both':
        np.fill_diagonal(np.flipud(arr_str), attr2)
        np.fill_diagonal(arr_str, attr1)

    # df style
    df_str = pd.DataFrame(arr_str, index=df.index, columns=df.columns)

    return df_str

#===== Style pandas dataframe =====
def style_rows(
    df: DataFrame,
    rows: Union[int,str,List,Tuple] =[-1],
    colors: Union[str,List,Tuple]="lightblue",
    )-> Styler:
    """Highlight rows in a dataframe.

    Parameters
    -----------
    df : DataFrame
        Pandas dataframe.
    rows : str or list of str
        Rows to highlight.
    colors : str or list or tuple
        Color of highlighted row.

    Examples
    ---------
    .. code-block:: python

        df = pd.DataFrame({'A':[10,20,30],
                    'B':['USA','Canada','Mexico'],
                    'C': [100,200,300]})
        bp.style_rows(df,1)
    """
    df_style = df.style.apply(_style_rows, axis=None,rows=rows,colors=colors)

    return df_style

def style_row_mi(
    df:DataFrame,
    name:Any=0,
    axis:Union[str,int]=1,
    color:SN=None,
    c:SN=None,
    )-> Styler:
    """Style rows of pandas dataframe.

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
                        if ser.name[0] == names[0][0]
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

def style_cols(
    df: DSt,
    cols: Union[str,int,List,Tuple] =[-1],
    colors: Union[str,List,Tuple] ="salmon",
    )-> Styler:
    """Highlight columns in a dataframe.

    Parameters
    -----------
    df : DataFrame or Styler
        Pandas dataframe.
    cols : str or list of str
        Columns to highlight.
    colors : str or list or tuple
        Color of highlighted column.

    Examples
    ---------
    .. code-block:: python

        df = pd.DataFrame({'A':[10,20,30],
                    'B':['USA','Canada','Mexico'],
                    'C': [100,200,300]})
        bp.style_cols(df,1)
    """
    df_style = df.style.apply(_style_cols, axis=None,cols=cols,colors=colors)
    return df_style
# alias
style_col = style_cols

def style_rowscols(
    df: DataFrame,
    rows: SIN =0,
    cols: SIN =0,
    c1: str ="lightblue",
    c2: str ="salmon",
    )-> Styler:
    return (
        df.style
        .apply(_style_rows, axis=None, colors=c1, rows=rows)
        .apply(_style_cols, axis=None, colors=c2, cols=cols)
    )
# aliases
style_rc = style_rowscols

def style_diags(
    df:DataFrame,
    diag:SI='both',
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
    return df.style.apply(_style_diags,axis=None,diag=diag,c1=c1,c2=c2)

def style_rowscolsdiags(
    df: DataFrame,
    rows: Union[str,int,List,Tuple] =[-1],
    cols: Union[str,int,List,Tuple] =[-1],
    diag:SI = 'both',
    c1: str ="lightblue",
    c2: str ="salmon",
    c3: str ="khaki"
    )-> Styler:
    return (
        df.style
        .apply(_style_rows, axis=None, colors=c1, rows=rows)
        .apply(_style_cols, axis=None, colors=c2, cols=cols)
        .apply(_style_diags, axis=None, c1=c3,c2=c3,diag=diag)
    )
# aliases
style_rcd = style_rowscolsdiags

def style_cellv(
    df:DataFrame,
    cond:str='',
    c:str='lightgreen',
    idx:SIN=None,
    col:SIN=None
    )-> Styler:
    """Style a cell element of a pandas dataframe using cell value.

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
    color:str='lightgreen'
    )-> Styler:
    """Style cell using dataframe a x and using condition.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cond: str
        Mask. eg. "x[x.index==2]==0"
    color: str
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
    def f(x:DSt,
        cond:str,
        color:str='lightgreen'
        )-> DataFrame:
        # if x is Styler, make it dataframe
        if isinstance(x,pd.io.formats.style.Styler):
            x = x.data

        attr = f'background-color: {color}'
        arr = np.full(x.shape, "", dtype="<U24")
        df_str = pd.DataFrame(arr, index=x.index,columns=x.columns)

        df_str[eval(cond)] = attr
        return df_str
    return df.style.apply(f,axis=None,cond=cond,color=color)