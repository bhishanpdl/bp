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
    "highlight_rcd",
    "describe",
    "make_date",
    "add_datepart",
    "add_elapsed_times",
    "cont_cat_split",
    "df_shrink_dtypes",
    "df_shrink",
]

from typing import List,Tuple,Dict,Union,Callable,Any,Sequence,Iterable,Optional
from pandas import DataFrame,Series


import numpy as np
import pandas as pd
import re

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

def highlight_row(ser: Series,
    color: str ="lightblue",
    row: str | int | None =None) -> List:
    if row is None:
        row = ser.index[-1]
    bkg = f"background-color: {color}"
    return [bkg if ser.name == row else "" for _ in ser]

def highlight_col(ser: Series,
    color: str ="salmon",
    col: str | int | None =None
    ) -> List:
    if col is None:
        col = ser.index[-1]
    bkg = f"background-color: {color}"
    return [bkg if ser.name == col else "" for _ in ser]

def highlight_diag(dfx: DataFrame, color: str ="khaki") -> DataFrame:
    a = np.full(dfx.shape, "", dtype="<U24")
    np.fill_diagonal(a, f"background-color: {color}")
    df1 = pd.DataFrame(a, index=dfx.index, columns=dfx.columns)
    return df1

def highlight_rowf(dfx : Dataframe,
    color :str ="lightblue",
    row: str | int | None =None
    ) -> Any:
    if row is None:
        row = dfx.index[-1]

    def highlight_row(ser: Series,
        row: str | int | None,
        color: str="salmon") -> List:
        bkg = f"background-color: {color}"
        return [bkg if ser.name == row else "" for _ in ser]

    return dfx.style.apply(highlight_row, axis=1, row=row)

def highlight_rowsf(dfx: DataFrame,
    color: str ="lightblue",
    rows: str | int | None =None
    ) -> Any:
    if rows is None:
        rows = [dfx.index[-1]]

    def highlight_row(ser: Series,
        row: str | int | None,
        color: str ="salmon"
        ) -> List:
        bkg = f"background-color: {color}"
        return [bkg if ser.name == row else "" for _ in ser]

    dfxs = dfx.copy()
    dfxs = dfxs.style.apply(highlight_row, axis=1, row=rows[0])

    for row in rows[1:]:
        dfxs = dfxs.apply(highlight_row, axis=1, row=row)

    return dfxs

def highlight_colf(dfx: DataFrame,
    color: str ="salmon",
    col: str | int | None =-1) -> Any:
    if not isinstance(col, str):
        col = dfx.columns[col]

    def highlight_col(dfy: DataFrame) -> DataFrame:  # axis=None needs dataframe
        bkg = f"background-color: {color}"
        df1 = pd.DataFrame("", index=dfy.index, columns=dfy.columns)
        df1.loc[:, col] = bkg
        return df1

    return dfx.style.apply(highlight_col, axis=None)

def highlight_diagf(dfx: DataFrame,
    color: str ="khaki"):
    def highlight_diag(dfy:DataFrame)->DataFrame:
        a = np.full(dfy.shape, "", dtype="<U24")
        np.fill_diagonal(a, f"background-color: {color}")
        df1 = pd.DataFrame(a, index=dfy.index, columns=dfy.columns)
        return df1

    return dfx.style.apply(highlight_diag, axis=None)

def highlight_rcd(dfx,
    row: str | int | None =None,
    col: str | int | None =None,
    c1: str ="lightblue",
    c2: str ="salmon",
    c3: str ="khaki"):
    return (
        dfx.style.apply(highlight_diag, axis=None, color=c3)
        .apply(highlight_row, axis=1, color=c1, row=row)
        .apply(highlight_col, axis=0, color=c2, col=col)
    )

def describe(df: DataFrame,
    cols: str | int | None =None,
    style: bool =True,
    print_: bool =False,
    sort_col: str | int ='Missing',
    transpose: bool =False,
    round_: int =2,
    fmt: str | None =None
    ):
    """Get nice table of columns description of given dataframe.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe.
    cols: list
        list of feature names
    style: bool
        Whether or not to style object or category data types.
    print_: bool
        Whether or not to print output dataframe
    sort_col: str
        Sort the output dataframe. eg. Missing, Unique, Type, Zeros
    transpose: bool
        Whether or not to transpose result.
    round_: int
        Rounding figures for floats.
    fmt: str
        String formatting for numbers. eg. "{:.2g}", "{:,.4f}"

    Usage
    ------
    .. code-block:: python

        from bhishan.util_ds import get_column_descriptions
        df.bp.describe(style=True)

    """
    # df = self._obj
    if cols is None:
        cols = df.columns
    df = df[cols]

    df_desc = pd.DataFrame()
    df_desc['Feature'] = df.columns
    df_desc['Feature2'] = df.columns
    df_desc['Type'] = df.dtypes.values
    df_desc['Missing'] = df.isnull().sum().values
    df_desc['N'] = len(df)
    df_desc['Count'] = len(df) - df_desc['Missing']
    df_desc['Zeros'] = df.eq(0).sum().values
    df_desc['Ones'] = df.eq(1).sum().values
    df_desc['Unique'] = df.nunique().values

    df_desc['MissingPct'] = df_desc['Missing'].div(len(df)).mul(100).round(2).values
    df_desc['ZerosPct'] = df_desc['Zeros'].div(len(df)).mul(100).round(2).values
    df_desc['OnesPct'] = df_desc['Ones'].div(len(df)).mul(100).round(2).values

    # extra columns
    df_desc['smallest5'] = [df[df.columns[i]].sort_values()[:5].tolist() for i in range(df.shape[1])]
    df_desc['largest5'] = [df[df.columns[i]].sort_values(ascending=False)[:5].tolist() for i in range(df.shape[1])]

    df_desc['first5'] = [df[df.columns[i]].iloc[:5].tolist() for i in range(df.shape[1])]
    df_desc['last5'] = [df[df.columns[i]].iloc[-5:].tolist() for i in range(df.shape[1])]

    df_desc = df_desc[['Feature', 'Feature2','Type', 'N','Count', 'Unique',
                        'Missing', 'MissingPct','Zeros','ZerosPct','Ones','OnesPct',
                        'smallest5','largest5','first5','last5'
                        ]]

    df_desc1 = df.describe().T # mean median etc describe
    df_desc1 = df_desc1[['mean','std','min','max','25%','50%','75%']]
    df_desc = df_desc.merge(df_desc1,left_on='Feature',right_index=True,how='left')

    # rearrange columns
    df_desc = df_desc[['Feature', 'Type', 'N','Count', 'Unique',
                        'Missing', 'MissingPct','Zeros','ZerosPct','Ones','OnesPct',
                        'mean','std','min','max','25%','50%','75%','Feature2',
                        'smallest5','largest5','first5','last5'
                        ]]

    # sorting
    if sort_col != 'index':
        if sort_col == 'Missing':
            df_desc = df_desc.sort_values(['Missing','Zeros'],ascending=False)
        else:
            df_desc = df_desc.sort_values(sort_col,ascending=False)

    # style
    cols_fmt = ['MissingPct','ZerosPct','OnesPct','mean','std',
                'min','max','25%','50%','75%']
    if fmt:
        myfmt = fmt
        fmt_dict = {i:myfmt for i in cols_fmt}
    else:
        myfmt = "{:." + str(round_) + "f}"
        fmt_dict = {i:myfmt for i in cols_fmt}

    if style:
        if transpose:
            for col in cols_fmt:
                df_desc[col] = df_desc[col].round(round_)

            df_desc_styled = (df_desc
                .T
                .astype(str).replace('nan','')
                .style
                .apply(lambda x: ["background: salmon"
                        if  str(v) in ['object','category']
                        else ""
                        for v in x], axis = 1)
                .apply(lambda dfx: ["background: salmon" if (str(v) == '1' and dfx.name == 'Unique')
                            else "" for v in dfx], axis = 1)
                )
        else:
            df_desc_styled = (df_desc.style
                .apply(lambda x: ["background: salmon"
                        if  str(v) in ['object','category']
                        else ""
                        for v in x], axis = 1)
                .apply(lambda dfx: ["background: salmon" if (str(v) == '1' and dfx.name == 'Unique')
                            else "" for v in dfx], axis = 0)
                .background_gradient(subset=['MissingPct','ZerosPct','OnesPct'])
                .format(fmt_dict,na_rep='')
                )

    if print_:
        print(df_desc)

    return df_desc_styled if style else df_desc

#============= Obtained from fastai.tabular.core
# https://github.com/fastai/fastai2/blob/master/fastai2/tabular/core.py

def make_date(df: DataFrame, date_field: str | int):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)

def add_datepart(dfx: DataFrame,
    col: str | int | None,
    prefix: str | None =None,
    drop: bool =True,
    time: bool =False,
    attr: List | Tuple | None =None
    ) -> DataFrame:
    "Helper function that adds columns relevant to a date in the column `col` of `df`."
    df = dfx.copy()
    make_date(df, col)
    field = df[col]
    prefix = ifnone(prefix, re.sub("[Dd]ate$", "", col))
    attr = [
        "Year",
        "Month",
        "Week",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
    ] if attr is None else attr
    if time:
        attr = attr + ["Hour", "Minute", "Second"]
    for n in attr:
        df[prefix + n] = getattr(field.dt, n.lower())
    df[prefix + "Elapsed"] = field.astype(np.int64) // 10 ** 9
    if drop:
        df.drop(col, axis=1, inplace=True)
    return df

def _get_elapsed(df: DataFrame,
    cols: List | Tuple,
    date_field: str | int,
    base_field: str | int,
    prefix: str | None) -> DataFrame:
    for f in cols:
        day1 = np.timedelta64(1, "D")
        last_date, last_base, res = np.datetime64(), None, []
        for b, v, d in zip(df[base_field].values, df[f].values, df[date_field].values):
            if last_base is None or b != last_base:
                last_date, last_base = np.datetime64(), b
            if v:
                last_date = d
            res.append(((d - last_date).astype("timedelta64[D]") / day1))
        df[prefix + f] = res
    return df

def add_elapsed_times(df: DataFrame,
    cols: str | List | Tuple,
    date_field: str | int,
    base_field: str | int,
    )-> DataFrame:
    "Add in `df` for each event in `cols` the elapsed time according to `date_field` grouped by `base_field`"
    cols = list(cols) if isinstance(cols, str) else cols
    # Make sure date_field is a date and base_field a bool
    df[cols] = df[cols].astype("bool")
    make_date(df, date_field)

    work_df = df[cols + [date_field, base_field]]
    work_df = work_df.sort_values([base_field, date_field])
    work_df = _get_elapsed(work_df, cols, date_field, base_field, "After")
    work_df = work_df.sort_values([base_field, date_field], ascending=[True, False])
    work_df = _get_elapsed(work_df, cols, date_field, base_field, "Before")

    for a in ["After" + f for f in cols] + ["Before" + f for f in cols]:
        work_df[a] = work_df[a].fillna(0).astype(int)

    for a, s in zip([True, False], ["_bw", "_fw"]):
        work_df = work_df.set_index(date_field)
        tmp = (
            work_df[[base_field] + cols]
            .sort_index(ascending=a)
            .groupby(base_field)
            .rolling(7, min_periods=1)
            .sum()
        )
        tmp.drop(base_field, 1, inplace=True)
        tmp.reset_index(inplace=True)
        work_df.reset_index(inplace=True)
        work_df = work_df.merge(tmp, "left", [date_field, base_field], suffixes=["", s])
    work_df.drop(cols, 1, inplace=True)
    return df.merge(work_df, how="left", on=[date_field, base_field])

def cont_cat_split(df: DataFrame,
    max_card: int =20,
    dep_var: str | int | None=None
    ) -> Tuple:
    "Helper function that returns column names of cont and cat variables from given `df`."
    cont_names, cat_names = [], []
    for label in df:
        if label == dep_var:
            continue
        if (
            df[label].dtype == int
            and df[label].unique().shape[0] > max_card
            or df[label].dtype == float
        ):
            cont_names.append(label)
        else:
            cat_names.append(label)
    return cont_names, cat_names

def df_shrink_dtypes(df: DataFrame,
    skip: List =[],
    obj2cat: bool =True,
    int2uint: bool =False
    ) -> Dict:
    "Return any possible smaller data types for DataFrame columns. Allows `object`->`category`, `int`->`uint`, and exclusion."

    # 1: Build column filter and typemap
    excl_types, skip = {"category", "datetime64[ns]", "bool"}, set(skip)

    typemap = {
        "int": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.int8, np.int16, np.int32, np.int64)
        ],
        "uint": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.uint8, np.uint16, np.uint32, np.uint64)
        ],
        "float": [
            (np.dtype(x), np.finfo(x).min, np.finfo(x).max)
            for x in (np.float32, np.float64, np.longdouble)
        ],
    }
    if obj2cat:
        typemap[
            "object"
        ] = "category"  # User wants to categorify dtype('Object'), which may not always save space
    else:
        excl_types.add("object")

    new_dtypes = {}
    exclude = lambda dt: dt[1].name not in excl_types and dt[0] not in skip

    for c, old_t in filter(exclude, df.dtypes.items()):
        t = next((v for k, v in typemap.items() if old_t.name.startswith(k)), None)

        if isinstance(t, list):  # Find the smallest type that fits
            if int2uint and t == typemap["int"] and df[c].min() >= 0:
                t = typemap["uint"]
            new_t = next(
                (r[0] for r in t if r[1] <= df[c].min() and r[2] >= df[c].max()), None
            )
            if new_t and new_t == old_t:
                new_t = None
        else:
            new_t = t if isinstance(t, str) else None

        if new_t:
            new_dtypes[c] = new_t
    return new_dtypes

def df_shrink(df: DataFrame,
    skip: List =[],
    obj2cat: bool =True,
    int2uint: bool =False
    ) -> DataFrame:
    "Reduce DataFrame memory usage, by casting to smaller types returned by `df_shrink_dtypes()`."
    dt = df_shrink_dtypes(df, skip, obj2cat=obj2cat, int2uint=int2uint)
    return df.astype(dt)
