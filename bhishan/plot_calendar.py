__author__ = 'Bhishan Poudel'

__doc__ = """
This module provides various datetime functions.

- print_calendar
- display_calendar

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    help(bp.ds_datetime)

"""

__all__ = [
    'print_calendar_month',
    'display_calendar_month',
    'display_calendar_month_red_green',
    'display_calendar_month_cmap'
    ]
from typing import List,Tuple,Dict,Callable,Iterable,Any,Union,Optional

import numpy as np
import pandas as pd
import calendar
import datetime
import matplotlib
from IPython.display import display,HTML

def print_calendar_month(y,m,d):
    """Print the calendar from year month day."""
    cal = calendar.TextCalendar(calendar.MONDAY)
    s = cal.formatmonth(y,m,d)
    print(s)

def display_calendar_month(
    y:int,
    m:int,
    d:Optional[int]=1,
    color: Union[str,bool,None] =None):
    """Display the calendar month.

    Parameters
    -----------
    y: int
        year
    m: int
        month
    color: str | bool | None
        color for given day of month

    Usage
    -------
    .. code-block:: python
        display_calendar_month(2021,7,color=True)
    """
    t = datetime.datetime(y,m,d)

    # create HTML Calendar month
    cal = calendar.HTMLCalendar()
    s = cal.formatmonth (t.year,t.month)

    # highlight given day
    if color is None:
        pass
    elif color == True:
        s = s.replace(f'>{t.day}<', f' bgcolor="darkgreen"><b><u>{t.day}</u></b><')
    else:
        s = s.replace(f'>{t.day}<', f' bgcolor="{color}"><b><u>{t.day}</u></b><')

    display(HTML(s))

def display_calendar_month_red_green(
    y:int,
    m:int,
    values:Iterable,
    indices:Union[Iterable,None]=None,
    integers:bool=False,
    red:str='lightcoral',
    green:str='darkseagreen',
    summary:bool=False,
    disp:bool=True,
    ):
    """Display the calendar month with green for positive values
    and red for negative values.

    Parameters
    -----------
    y: int
        year
    m: int
        month
    values: Iterable
        value for each day of month
    integers: bool
        Make all values integers.
    indices: Iterable
        Index for each value, e.g. [1,3,5] for only these days of months.
    red: str
        color for negative values
    green: str
        color for positive values
    summary: bool
        if True, display summary
    disp: bool
        if True, display html in jupyter notebook

    Usage
    -------
    .. code-block:: python

        values = np.random.randint(-100,100,size=20)
        display_calendar_month_red_green(2021,7,values)
    """

    # get datetime
    cal = calendar.HTMLCalendar()
    t = datetime.datetime(y,m,1)
    s = cal.formatmonth (t.year,t.month)

    # case when indices is None
    if indices is None:
        indices = list(range(len(values)))

    # make values integers
    if integers:
        values = list(map(int, values))

    # change color of cells red or green
    for i,v in zip(indices,values):
        c = red if v <= 0 else green
        s = s.replace(f'>{i}<', f' bgcolor="{c}"><b><u>{i} ({v})</u></b><')

    # summary
    num_pos_days = sum(1 for i in values if i >0)
    num_neg_days = sum(1 for i in values if i <= 0)

    # avoid zero division
    num_pos_days = num_pos_days if num_pos_days > 0 else 1
    num_neg_days = num_neg_days if num_neg_days > 0 else 1

    avg_pos_days = sum(i for i in values if i >0) / num_pos_days
    avg_neg_days = sum(i for i in values if i <= 0) / num_neg_days
    total = np.sum(values)

    repl = f"""
        Summary for {t.strftime("%b %Y")}
        =================
        +ve days: {num_pos_days}
        -ve days: {num_neg_days}

        Avg +ve : {avg_pos_days:.0f}
        Avg -ve : {avg_neg_days:.0f}

        Total: {total:.0f} </br>
    """
    lst = repl.split('\n')
    repl = '</br>'.join(lst)

    if summary:
        month = t.strftime("%B %Y")
        s = s.replace(month,repl)

    # display html in jupyter notebook
    if disp:
        display(HTML(s))
    else:
        return s

def display_calendar_month_cmap(
    y:int,
    m:int,
    cmap_name:str='Reds_r',
    indices:Union[Iterable,None]=None,
    values:Union[Iterable,None] = None,
    ):
    """Display the calendar month using cmap for values.

    Parameters
    -----------
    y: int
        year
    m: int
        month
    values: Iterable
        value for each day of month
    indices: Iterable
        Index for each value, e.g. [1,3,5] for only these days of months.
    cmap_name: str
        name of colormap eg. Reds_r, Greens

    Usage
    -------
    .. code-block:: python

        values = np.random.randint(-100,100,size=20)
        display_calendar_month_cmap(2021,7,values=values)
    """

    # get datetime
    cal = calendar.HTMLCalendar()
    t = datetime.datetime(y,m,1)
    s = cal.formatmonth (t.year,t.month)

    # case when values is None
    if values is None:
        return display(HTML(s))

    # case when indices is None
    if indices is None:
        indices = list(range(len(values)))

    # make values series
    ser_values = pd.Series(values)

    # get color names
    cmap = matplotlib.cm.get_cmap(cmap_name, len(values))
    colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    ser_colors = pd.Series(colors)
    colors = ser_values.index.map(ser_colors).fillna('')

    # change color of cells
    for i,day in enumerate(indices):
        c = colors[i]
        v = values[i]
        s = s.replace(f'>{day}<', f' bgcolor="{c}"><b><u>{day} ({v})</u></b><')

    # display html in jupyter notebook
    return display(HTML(s))

