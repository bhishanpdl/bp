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

__all__ = ['print_calendar','display_calendar']

import numpy as np
import pandas as pd
import calendar
import datetime
from IPython.display import display,HTML

def print_calendar(y,m,d):
    """Print the calendar from year month day."""
    c = calendar.TextCalendar(calendar.MONDAY)
    s = c.formatmonth(y,m,d)
    print(s)

def display_calendar(y,m,d):
    """Display the calendar from year month day."""
    t = datetime.datetime(y,m,d)

    # create HTML Calendar month
    cal = calendar.HTMLCalendar()
    s = cal.formatmonth (t.year,t.month)

    # highlight given day
    ss = s.replace('>%i<'%t.day, ' bgcolor="#66ff66"><b><u>%i</u></b><'%t.day)
    display(HTML(ss))





