__author__ = 'Bhishan Poudel'

__doc__ = """
This package is my personal library. It contains various tools for performing
day to day data analysis. It contains data cleaning, machine learning, and data
visualization helper functions.

On top of that it also has incorporated pandas api extension, so that I can
use various data frame accessor operation right on top of top-level pandas
dataframe.

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.show_method_attributes(bp)
    help(bp)

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Note: use __all__ variable on top of modules to determine what to
# display when we use import *

# Import everything
from bhishan.utils import *
from bhishan.ds_ds import *
from bhishan.ds_json import *
from bhishan.ds_speed import *
from bhishan.ds_stats import *
from bhishan.ml_model_eval import *
from bhishan.ml_data_preprocessing import *
from bhishan.ml_statsmodels import *
from bhishan.plot_utils import *
from bhishan.plot_colors import *
from bhishan.plot_plotly import *
from bhishan.plot_map import *
from bhishan.plot_ds import *
from bhishan.plot_model_eval import *
from bhishan.plot_model_eval import *
from bhishan.plot_model import *
from bhishan.hlp import *

# Not used modules
# from bhishan.plot_bokeh import *








