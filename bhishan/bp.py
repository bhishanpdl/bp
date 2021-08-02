__author__ = 'Bhishan Poudel'

__doc__ = """
This package contains various data science functions and adds api for pandas dataframe.
It provides functionalities for machine learning, statistics, data analysis, and data visualization.

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.show_methods(bp)
    help(bp)

"""

# Note: use __all__ variable on top of all .py files
# to determine what to display when we use import *

#from bhishan.bp_details import * # this is internal file
from bhishan.ds_corr_outliers import *
from bhishan.ds_ds import *
from bhishan.ds_json import *
from bhishan.ds_speed import *
from bhishan.hlp import *
from bhishan.ml_data_proc import *
from bhishan.ml_modelling import *
from bhishan.ml_statsmodels import *
from bhishan.pandas_api import *
from bhishan.plot_calendar import *
from bhishan.plot_corr import *
from bhishan.plot_ds import *
from bhishan.plot_map import *
from bhishan.plot_modelling import *
from bhishan.plot_num_cat import *
from bhishan.plot_plotly import *
from bhishan.plot_stats import *
from bhishan.plot_tsa import *
from bhishan.util_colors import *
from bhishan.util_pd_styles import *
from bhishan.util_pd import *
from bhishan.util_plots import *
from bhishan.utils import *




