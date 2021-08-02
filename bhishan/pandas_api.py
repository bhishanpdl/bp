from __future__ import annotations
# this must be the first line.

__author__ = 'Bhishan Poudel'

__doc__ = """
This module extends the pandas dataframe API and adds various attributes.

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp?

"""
__all__ = ["BPAccessor"]

# type hints
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
from typing import Optional, Sequence, Type, TypeVar
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame, Series
from pandas.io.formats.style import Styler
try:
    from .mytyping import (IN, SN, SI, SIN, TL, LD, TLN, LDN,
    DS, DSt, NUM, NUMN, AD, AS, DN,
    ARR, ARRN, SARR, SARRN, LIMIT, LIMITN,
    LTii,LTss,LTff,LTsi,
    )
except:
    from mytyping import (IN, SN, SI, SIN, TL, LD, TLN, LDN,
    DS, DSt, NUM, NUMN, AD, AS, DN,
    ARR, ARRN, SARR, SARRN, LIMIT, LIMITN,
    LTii,LTss,LTff,LTsi,
    )
import numpy as np
import pandas as pd

import os
from functools import reduce

from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype

import scipy
from scipy import stats
from scipy.stats.morestats import binom_test

import seaborn as sns
sns.set(color_codes=True)
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot') # better than sns styles.
from matplotlib.ticker import PercentFormatter

import plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.tools as tls
from plotly.offline import plot, iplot
from plotly.subplots import make_subplots

import matplotlib as mpl
fontsize = 14
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['axes.titlesize'] = fontsize + 2
mpl.rcParams['axes.labelsize'] = fontsize

try:
    from .util_plots import (add_text_barplot, magnify,
                        get_mpl_style, get_plotly_colorscale)
except:
    from util_plots import (add_text_barplot, magnify,
                        get_mpl_style, get_plotly_colorscale)

@pd.api.extensions.register_dataframe_accessor("bp")
class BPAccessor:
    """Pandas dataframe accessor for various helper functions.

    Adapted from Original Module
    ============================
    - freq(cols,thres=1) Frequency statistics of columns
    - subtotal Get subtotals

    Descriptions
    ==============
    - describe(column_list=None,style=False)
    - missing() Missing values info.
    - get_duplicate_columns
    - value_counts

    Plotting Numerical and Categorical Features
    ============================================
    - plot_xxx num cat num_num num_cat cat_num cat_cat

    Timeseries Plots
    =================
    - plot_date_cat('date','target',save=True,show=False)
    - plot_daily_cat('date','subscribed')

    Statistics
    ===========
    - plot_corr(cols=None)
    - plot_corr_style(cols=None)
    - get_most_correlated_features(print_=False,thresh=0.5)
    - partial_corr(cols,print_=False)
    - plot_statistics(cols,statistic,color)

    - outliers_tukey
    - outliers_kde

    - compare_kde_binn

    Miscellaneous Plots
    ====================
    - plot_boxplot_cats_num(cats,num)
    - plot_count_cat(cat, percent=True)
    - plot_pareto

    Plotly Plots
    =============
    - plotly_countplot(col,topN=None)
    - plotly_corr_heatmap(target,topN=10)

    """
    SORT_FLAG = '~~~~zz'
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify this is a DataFrame
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a pandas DataFrame")

    #=========== Helper Functions ========================
    def _annotate_pearsonr(self, x, y, **kws):
        from scipy import stats
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("pearsonr = {:.2f}".format(r),
                    xy=(.1, .9), xycoords=ax.transAxes)

    def freq(self,
            cols,
            thresh=1,
            other_label='Others',
            clip_0=True,
            value=None,
            style=False):
        """ Create a table that counts the frequency of occurrence or summation of values
        for one or more columns of data. Table is sorted and includes cumulative
        values which can be useful for identifying a cutoff.

        Example of Titanic df.bhishan.freq(['class']):
        	        class	Count	Percent	    Cumulative Count	Cumulative Percent
                0	Third	491	    0.551066	491	                0.551066
                1	First	216	    0.242424	707	                0.793490
                2	Second	184	    0.206510	891	                1.000000

        Args:
            cols (list):       dataframe column names that will be grouped together
            thresh (float):    all values after this percentage will be combined into a
                            single category. Default is to not cut any values off
            other_label (str): if cutoff is used, this text will be used in the dataframe results
            clip_0 (bool):     In cases where 0 counts are generated, remove them from the list
            value (str):       Column that will be summed. If provided, summation is done
                            instead of counting each entry
            style (bool):     Apply a pandas style to format percentages

        Returns:
            Dataframe that summarizes the number of occurrences of each value in the provided
            columns or the sum of the data provided in the value parameter

        """
        if not isinstance(cols, list):
            raise AttributeError('Must pass a list of columns')

        if isinstance(value, list):
            raise AttributeError('value must be a string not a list')

        if value and value not in self._obj.columns:
            raise AttributeError('value must be a column name')

        if value and not is_numeric_dtype(self._obj[value]):
            raise AttributeError(f'{value} must be a numeric column')

        if thresh > 1:
            raise AttributeError('Cutoff must be <= 1.0')

        # Determine aggregation (counts or summation) for each item in column

        # TODO: NaNs need to be handled better. Wait for pandas 1.1
        # https://pandas.pydata.org/pandas-docs/dev/whatsnew/v1.1.0.html#allow-na-in-groupby-key
        if value:
            col_name = value
            agg_func = {value: 'sum'}
            group_data = self._obj.groupby(cols).agg(agg_func).reset_index()
        else:
            col_name = 'Count'
            group_data = self._obj.groupby(cols).size().reset_index(
                name=col_name)

        # Sort the results and cleanup the index
        results = group_data.sort_values(
            [col_name] + cols, ascending=False).reset_index(drop=True)

        # In data with null values, can include 0 counts filter them out by default
        if clip_0:
            results = results[results[col_name] > 0]

        # Include percents
        total = results[col_name].sum()
        results['Percent'] = results[col_name] / total

        # Keep track of cumulative counts or totals as well as their relative percent
        results[f'Cumulative {col_name}'] = results[col_name].cumsum()
        results[f'Cumulative Percent'] = results[f'Cumulative {col_name}'] / total

        # cutoff is a percentage below which all values are grouped together in an
        # others category
        if thresh < 1:
            # Flag the All Other rows
            results['Others'] = False
            results.loc[results[f'Cumulative Percent'] > thresh,
                        'Others'] = True

            # Calculate the total amount and percentage of the others
            other_total = results.loc[results['Others'], col_name].sum()
            other_pct = other_total / total

            # Create the footer row to append to the results
            all_others = pd.DataFrame({
                col_name: [other_total],
                'Percent': [other_pct],
                f'Cumulative {col_name}': [total],
                'Cumulative Percent': [1.0]
            })

            # Add the footer row, remove the Others column and rename the placeholder
            results = results[results['Others'] == False].append(
                all_others, ignore_index=True).drop(columns=['Others']).fillna(
                    dict.fromkeys(cols, other_label))

        if style:
            format_dict = {
                'Percent': '{:.2%}',
                'Cumulative Percent': '{:.2%}',
                'Count': '{0:,.0f}',
                f'{col_name}': '{0:,.0f}',
                f'Cumulative {col_name}': '{0:,.0f}'
            }
            return results.style.format(format_dict)
        else:
            return results

    def _get_group_levels(self, level=1):
        """Internal helper function to flatten out the group list from a multiindex

        Args:
            level (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """
        list_items = [col[0:level] for col in self._obj.index]
        results = []
        for x in list_items:
            if x not in results:
                results += [x]
        return results

    def _clean_labels(self, multi_index):
        """ Remove flags on the subtotal labels that are used to enforce sorting. This is an internal function

        Args:
            multi_index (pandas multi-index): Multi Index that
            includes the subtotal ordering text
        """
        master_list = []
        names = list(multi_index.names)
        for index_item in multi_index:
            sub_list = []
            for level in index_item:
                if level.startswith(self.SORT_FLAG):
                    level_val = level[len(self.SORT_FLAG):]
                else:
                    level_val = level
                sub_list.append(level_val)
            master_list.append(tuple(sub_list))
        return pd.MultiIndex.from_tuples(tuple(master_list), names=names)
    def _calc_subtotal(self, sub_level=None, sub_label='subtotal',
            show_sep=True, sep=' | '):
        """ Internal helper function to calculate one level of subtotals. Do not call directly.

            sub_level (int):       Grouping level to calculate subtotal.
                                    Default is max available.
            sub_label (str):       Label override for the sub total of the group
            show_sep  (bool):      Default is True to show subtotal levels
                                    separated by one or more characters
            sep (str):             Seperator for levels, defaults to |

        Returns:
            DataFrame Sub Total
        """

        all_levels = self._obj.index.nlevels
        output = []
        # Get the total for each cross section of the multi-index
        for cross_section in self._get_group_levels(sub_level):
            # Need to have blank spaces in label names so that all results will
            # line up correctly
            num_spaces = all_levels - len(cross_section)
            if show_sep:
                total_label = self.SORT_FLAG + sep.join(
                    cross_section) + f' - {sub_label}'
            else:
                total_label = self.SORT_FLAG + f'{sub_label}'
            sub_total_label = list(cross_section[0:sub_level]) + [
                total_label
            ] + [' '] * num_spaces
            # Pull out the actual section and total it
            section = self._obj.xs(cross_section, drop_level=False)
            subtotal = section.sum(numeric_only=True).rename(
                tuple(sub_total_label))
            output.append(section.append(subtotal))

        return pd.concat(output)

    def subtotal(self, sub_level=None, grand_label='grand_total',
                sub_label='subtotal', show_sep=True, sep=' | ',
                style=True,fmt="{:,.2f}"
                ):
        """ Add a numeric subtotals to a DataFrame.

        If the DataFrame has a multi-index, will
        add a subtotal at all levels defined in sub_level
        as well as a Grand Total

            sub_level (int or list): Grouping level to calculate subtotal.
                                    Default is max available.
                                    Can pass a single integer or a list
                                    of valid levels.
            grand_label (str):       Label override for the total of the entire DataFrame
            sub_label (str):         Label override for the sub total of the group
            show_sep  (bool):        Default is True to show subtotal levels
                                    separated by one or more characters
            sep (str):               Seperator for levels, defaults to |
            style (bool):            Style subtotal
            fmt (str):               Format of number.

        Returns:
            DataFrame with Grand Total and Sub Total levels as specified in sub_level
        """
        all_levels = self._obj.index.nlevels

        # Validate seperator is a string
        if not isinstance(sep, str):
            raise AttributeError('sep must be a string')
        # No value is specified, use the maximum
        if sub_level is None:
            sub_calc_list = list(range(1, all_levels))
        # Sort the list
        elif isinstance(sub_level, list):
            sub_calc_list = sub_level
            sub_calc_list.sort()
        # Convert an integer to a list
        elif isinstance(sub_level, int):
            sub_calc_list = [sub_level]

        grand_total_label = tuple([f'{grand_label}'] +
                                [' ' for _ in range(1, all_levels)])

        # If this is not a multiindex, add the grand total to the DataFrame
        if all_levels == 1:
            # No subtotals since no groups
            # Make sure the index is an object so we can append the subtotal without
            # Getting into Categorical issues
            self._obj.index = self._obj.index.astype('object')

            # If not multi-level, rename should not be a tuple
            # Add the Grand Total label at the end
            return self._obj.append(
                self._obj.sum(numeric_only=True).rename(grand_total_label[0]))

        # Check that list is in the appropriate range
        if sub_calc_list[0] <= 0 or sub_calc_list[-1] > all_levels - 1:
            raise AttributeError(
                f'Subtotal level must be between 1 and {all_levels-1}')

        # Remove any categorical indices
        self._obj.index = pd.MultiIndex.from_tuples(
            [n for i, n in enumerate(self._obj.index)],
            names=list(self._obj.index.names))

        subtotal_levels = []
        # Calculate the subtotal at each level given
        for i in sub_calc_list:
            level_result = self._calc_subtotal(sub_level=i,
                                            sub_label=sub_label,
                                            show_sep=show_sep,
                                            sep=sep)
            subtotal_levels.append(level_result)

        # Use combine first to join all the individual levels together into a single
        # DataFrame
        results = reduce(lambda l, r: l.combine_first(r), subtotal_levels)

        # Remove the subtotal sorting values
        results.index = self._clean_labels(results.index)

        # Final step is to add Grand total
        out = results.append(
            self._obj.sum(numeric_only=True).rename(grand_total_label))

        if style:
            out = (out.style
            .apply(lambda ser: ['background: lightblue'
                            if ser.name[1].endswith('subtotal') else ''
                            for _ in ser],axis=1)
            .apply(lambda ser: ['background: salmon'
                            if ser.name[0] == 'grand_total' else ''
                            for _ in ser],axis=1)
            .format(fmt,na_rep='')
            )

        return out


#============================= My extension ================================
# description
try:
    from .util_pd import describe
except:
    from util_pd import describe

BPAccessor.describe = describe
