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
from functools import wraps
def make_method(f):
    @wraps(f)
    def _(self, *args, **kwargs):
        return f(self._obj, *args, **kwargs)
    return _

# ds_corr_outliers
try:
    from .ds_corr_outliers import (
    corrwith,
    corr_high,
    corr_high_lst,
    partial_corr,
    point_biserial_correlation,
    outliers_tukey,
    outliers_kde
    )
except:
    from ds_corr_outliers import (
    corrwith,
    corr_high,
    corr_high_lst,
    partial_corr,
    point_biserial_correlation,
    outliers_tukey,
    outliers_kde
    )

BPAccessor.corrwith = make_method(corrwith)
BPAccessor.corr_high = make_method(corr_high)
BPAccessor.corr_high_lst = make_method(corr_high_lst)
BPAccessor.partial_corr = make_method(partial_corr)
BPAccessor.point_biserial_correlation = make_method(point_biserial_correlation)
BPAccessor.outliers_tukey = make_method(outliers_tukey)
BPAccessor.outliers_kde = make_method(outliers_kde)

# ds_ds
try:
    from .ds_ds import (
    freq_count,
    get_column_descriptions,
    report_cat_binn,
    compare_kde_binn,
    compare_kde2
    )
except:
    from ds_ds import (
    freq_count,
    get_column_descriptions,
    report_cat_binn,
    compare_kde_binn,
    compare_kde2
    )

BPAccessor.freq_count = make_method(freq_count)
BPAccessor.get_column_descriptions = make_method(get_column_descriptions)
BPAccessor.report_cat_binn = make_method(report_cat_binn)
BPAccessor.compare_kde_binn = make_method(compare_kde_binn)
BPAccessor.compare_kde2 = make_method(compare_kde2)

# ds_json
try:
    from .ds_json import parse_json_col
except:
    from ds_json import parse_json_col

BPAccessor.parse_json_col = make_method(parse_json_col)

# ds_speed
try:
    from .ds_speed import optimize_memory
except:
    from ds_speed import optimize_memory

BPAccessor.optimize_memory = make_method(optimize_memory)

# ml_data_proc
try:
    from .ml_data_proc import (
        get_outliers ,
        get_outliers_iqr,
        get_outliers_tukey ,
        get_outliers_kde ,
        remove_outliers ,
        remove_outliers_iqr ,
        remove_outliers_tukey ,
        add_interactions ,
    )
except:
    from ml_data_proc import (
        get_outliers ,
        get_outliers_iqr,
        get_outliers_tukey ,
        get_outliers_kde ,
        remove_outliers ,
        remove_outliers_iqr ,
        remove_outliers_tukey ,
        add_interactions ,
    )

BPAccessor.get_outliers = make_method(get_outliers)
BPAccessor.get_outliers_iqr = make_method(get_outliers_iqr)
BPAccessor.get_outliers_tukey = make_method(get_outliers_tukey)
BPAccessor.get_outliers_kde = make_method(get_outliers_kde)
BPAccessor.remove_outliers = make_method(remove_outliers)
BPAccessor.remove_outliers_iqr = make_method(remove_outliers_iqr)
BPAccessor.remove_outliers_tukey = make_method(remove_outliers_tukey)
BPAccessor.add_interactions = make_method(add_interactions)

# ml_modelling (just use functions)
# ml_statsmodels (just use functions)
# plot_calendar (just use functions)

# plot_corr
try:
    from .plot_corr import (
        plot_corr ,
        plot_corr_style ,
        plot_corr_sns ,
        plot_corrplot_with_pearsonr ,
        plot_multiple_jointplots_with_pearsonr
    )
except:
    from plot_corr import (
        plot_corr ,
        plot_corr_style ,
        plot_corr_sns ,
        plot_corrplot_with_pearsonr,
        plot_multiple_jointplots_with_pearsonr
    )

BPAccessor.plot_corr = make_method(plot_corr)
BPAccessor.plot_corr_style = make_method(plot_corr_style)
BPAccessor.plot_corr_sns = make_method(plot_corr_sns)
BPAccessor.plot_corrplot_with_pearsonr = make_method(plot_corrplot_with_pearsonr)
BPAccessor.plot_multiple_jointplots_with_pearsonr = make_method(plot_multiple_jointplots_with_pearsonr)

# plot_ds
try:
    from .plot_ds import (
        plot_stem,
        plot_pareto,
        countplot,
        regplot_binn,
        plot_two_clusters,
    )
except:
    from plot_ds import (
        plot_stem,
        plot_pareto,
        countplot,
        regplot_binn,
        plot_two_clusters,
    )

BPAccessor.plot_stem = make_method(plot_stem)
BPAccessor.plot_pareto = make_method(plot_pareto)
BPAccessor.countplot = make_method(countplot)
BPAccessor.regplot_binn = make_method(regplot_binn)
BPAccessor.plot_two_clusters = make_method(plot_two_clusters)

# plot_map
try:
    from .plot_map import (
    plotly_usa_map,
    plotly_usa_map2,
    plotly_usa_map_agg,
    plotly_usa_map_bubble,
    plotly_country_map,
    plotly_country_map_agg,
    plotly_mapbox
    )
except:
    from plot_map import (
    plotly_usa_map,
    plotly_usa_map2,
    plotly_usa_map_agg,
    plotly_usa_map_bubble,
    plotly_country_map,
    plotly_country_map_agg,
    plotly_mapbox
    )
BPAccessor.plotly_usa_map = make_method(plotly_usa_map)
BPAccessor.plotly_usa_map2 = make_method(plotly_usa_map2)
BPAccessor.plotly_usa_map_agg = make_method(plotly_usa_map_agg)
BPAccessor.plotly_usa_map_bubble = make_method(plotly_usa_map_bubble)
BPAccessor.plotly_country_map = make_method(plotly_country_map)
BPAccessor.plotly_country_map_agg = make_method(plotly_country_map_agg)
BPAccessor.plotly_mapbox = make_method(plotly_mapbox)

# plot_modelling (just use functions)
# plot_num_cat
try:
    from .plot_num_cat import (
    plot_num,
    plot_cat,
    plot_num_num,
    plot_num_cat,
    plot_cat_num,
    plot_cat_cat,
    plot_cat_stacked,
    plot_boxplot_cats_num,
    plot_count_cat,
    plot_cat_cat2,
    plot_num_cat2,
    plot_cat_binn,
    plot_cat_cat_pct,
    plot_donut_binn,
    )
except:
    from plot_num_cat import (
    plot_num,
    plot_cat,
    plot_num_num,
    plot_num_cat,
    plot_cat_num,
    plot_cat_cat,
    plot_cat_stacked,
    plot_boxplot_cats_num,
    plot_count_cat,
    plot_cat_cat2,
    plot_num_cat2,
    plot_cat_binn,
    plot_cat_cat_pct,
    plot_donut_binn,
    )
BPAccessor.plot_num = make_method(plot_num)
BPAccessor.plot_cat = make_method(plot_cat)
BPAccessor.plot_num_num = make_method(plot_num_num)
BPAccessor.plot_num_cat = make_method(plot_num_cat)
BPAccessor.plot_cat_num = make_method(plot_cat_num)
BPAccessor.plot_cat_cat = make_method(plot_cat_cat)
BPAccessor.plot_cat_stacked = make_method(plot_cat_stacked)
BPAccessor.plot_boxplot_cats_num = make_method(plot_boxplot_cats_num)
BPAccessor.plot_count_cat = make_method(plot_count_cat)
BPAccessor.plot_cat_cat2 = make_method(plot_cat_cat2)
BPAccessor.plot_num_cat2 = make_method(plot_num_cat2)
BPAccessor.plot_cat_binn = make_method(plot_cat_binn)
BPAccessor.plot_cat_cat_pct = make_method(plot_cat_cat_pct)
BPAccessor.plot_donut_binn = make_method(plot_donut_binn)

# plot_plotly
try:
    from .plot_plotly import (
    plotly_corr,
    plotly_corr_heatmap,
    plotly_countplot,
    plotly_histogram,
    plotly_distplot,
    plotly_radar_plot,
    plotly_boxplot,
    plotly_boxplot_allpoints_with_outliers,
    plotly_boxplot_categorical_column,
    plotly_scattergl_plot,
    plotly_scattergl_plot_colorcol,
    plotly_scattergl_plot_subplots,
    plotly_bubbleplot
    )
except:
    from plot_plotly import (
    plotly_corr,
    plotly_corr_heatmap,
    plotly_countplot,
    plotly_histogram,
    plotly_distplot,
    plotly_radar_plot,
    plotly_boxplot,
    plotly_boxplot_allpoints_with_outliers,
    plotly_boxplot_categorical_column,
    plotly_scattergl_plot,
    plotly_scattergl_plot_colorcol,
    plotly_scattergl_plot_subplots,
    plotly_bubbleplot
    )
BPAccessor.plotly_corr = make_method(plotly_corr)
BPAccessor.plotly_corr_heatmap = make_method(plotly_corr_heatmap)
BPAccessor.plotly_countplot = make_method(plotly_countplot)
BPAccessor.plotly_histogram = make_method(plotly_histogram)
BPAccessor.plotly_distplot = make_method(plotly_distplot)
BPAccessor.plotly_radar_plot = make_method(plotly_radar_plot)
BPAccessor.plotly_boxplot = make_method(plotly_boxplot)
BPAccessor.plotly_boxplot_allpoints_with_outliers = make_method(plotly_boxplot_allpoints_with_outliers)
BPAccessor.plotly_boxplot_categorical_column = make_method(plotly_boxplot_categorical_column)
BPAccessor.plotly_scattergl_plot = make_method(plotly_scattergl_plot)
BPAccessor.plotly_scattergl_plot_colorcol = make_method(plotly_scattergl_plot_colorcol)
BPAccessor.plotly_scattergl_plot_subplots = make_method(plotly_scattergl_plot_subplots)
BPAccessor.plotly_bubbleplot = make_method(plotly_bubbleplot)

# plot_stats
try:
    from .plot_stats import(
    plot_statistics,
    plot_ecdf,
    plot_gini,
    plot_ks,
    get_yprobs_sorted_proportions,
    )
except:
    from plot_stats import(
    plot_statistics,
    plot_ecdf,
    plot_gini,
    plot_ks,
    get_yprobs_sorted_proportions,
    )
BPAccessor.plot_statistics = make_method(plot_statistics)
BPAccessor.plot_ecdf = make_method(plot_ecdf)
BPAccessor.plot_gini = make_method(plot_gini)
BPAccessor.plot_ks = make_method(plot_ks)
BPAccessor.get_yprobs_sorted_proportions = make_method(get_yprobs_sorted_proportions)

# plot_tsa
try:
    from .plot_tsa import plot_date_cat,plot_daily_cat
except:
    from plot_tsa import plot_date_cat,plot_daily_cat
BPAccessor.plot_date_cat = make_method(plot_date_cat)
BPAccessor.plot_daily_cat = make_method(plot_daily_cat)

# util_colors (just use functions)

# util_pd_styles
try:
    from .util_pd_styles import (
    style_rows,
    style_cols,
    style_row_mi,
    style_diags,
    style_rowscols,
    style_rc,
    style_rowscolsdiags,
    style_rcd,
    style_cellv,
    style_cellx,
    )
except:
    from util_pd_styles import (
    style_rows,
    style_cols,
    style_row_mi,
    style_diags,
    style_rowscols,
    style_rc,
    style_rowscolsdiags,
    style_rcd,
    style_cellv,
    style_cellx,
    )
BPAccessor.style_rows = make_method(style_rows)
BPAccessor.style_cols = make_method(style_cols)
BPAccessor.style_row_mi = make_method(style_row_mi)
BPAccessor.style_diags = make_method(style_diags)
BPAccessor.style_rowscols = make_method(style_rowscols)
BPAccessor.style_rc = make_method(style_rc)
BPAccessor.style_rowscolsdiags = make_method(style_rowscolsdiags)
BPAccessor.style_rcd = make_method(style_rcd)
BPAccessor.style_cellv = make_method(style_cellv)
BPAccessor.style_cellx = make_method(style_cellx)

# util pd
try:
    from .util_pd import (
        describe,
        add_datepart,
        df_shrink
    )
except:
    from util_pd import (
        describe,
        add_datepart,
        df_shrink
    )
BPAccessor.describe = make_method(describe)
BPAccessor.add_datepart = make_method(add_datepart)
BPAccessor.df_shrink = make_method(df_shrink)





