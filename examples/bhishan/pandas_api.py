# -*- coding: utf-8 -*-
from __future__ import annotations
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

from .plot_utils import (add_text_barplot, magnify,
                        get_mpl_style, get_plotly_colorscale)

from typing import Tuple, List, Dict
from typing import Any, Optional, Sequence, Union, Type, TypeVar

@pd.api.extensions.register_dataframe_accessor("bp")
class BPAccessor:
    """Pandas dataframe accessor for various helper functions.

    Descriptions
    ==============
    - freq(cols,thres=1) Frequency statistics of columns
    - missing() Missing values info.
    - get_column_descriptions(column_list=None,style=False)

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
    - corrplot_with_pearsonr(cols)
    - get_most_correlated_features(print_=False,thresh=0.5)
    - partial_corr(cols,print_=False)

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
    def describe(self, cols=None,style=True,print_=False,
                sort_col='Missing',transpose=False,round_=2,fmt=None):
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
        df = self._obj
        if cols is None:
            cols = df.columns
        df = df[cols]

        df_desc = pd.DataFrame()
        df_desc['Feature'] = df.columns
        df_desc['Type'] = df.dtypes.values
        df_desc['Missing'] = df.isnull().sum().values
        df_desc['N'] = len(df)
        df_desc['Count'] = len(df) - df_desc['Missing']
        df_desc['Zeros'] = df.eq(0).sum().values
        df_desc['Unique'] = df.nunique().values

        df_desc['MissingPct'] = df_desc['Missing'].div(len(df)).mul(100).round(2).values
        df_desc['ZerosPct'] = df_desc['Zeros'].div(len(df)).mul(100).round(2).values

        df_desc = df_desc[['Feature', 'Type', 'N','Count', 'Unique',
                            'Missing', 'MissingPct','Zeros','ZerosPct']]

        df_desc1 = df.describe().T
        df_desc1 = df_desc1[['mean','std','min','max','25%','50%','75%']]
        df_desc = df_desc.merge(df_desc1,left_on='Feature',right_index=True,how='left')

        # sorting
        if sort_col != 'index':
            if sort_col == 'Missing':
                df_desc = df_desc.sort_values(['Missing','Zeros'],ascending=False)
            else:
                df_desc = df_desc.sort_values(sort_col,ascending=False)

        # style
        cols_fmt = ['MissingPct','ZerosPct','mean','std',
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
                    )
            else:
                df_desc_styled = (df_desc.style
                    .apply(lambda x: ["background: salmon"
                            if  str(v) in ['object','category']
                            else ""
                            for v in x], axis = 1)
                    .background_gradient(subset=['MissingPct','ZerosPct'])
                    .format(fmt_dict,na_rep='')
                    )

        if print_:
            print(df_desc)

        return df_desc_styled if style else df_desc

    def get_duplicate_columns(self,print_=True):
        """Get a list of duplicated columns.

        The function first group the dataframe by dtypes. If the data type
        if category it is converted into object data type.
        Then it will loop through all the columns to check if any two columns
        are equal and returns the column names of duplcated columns.

        Parameters
        -----------
        df: pandas.DataFrame
            Input pandas dataframe.
        print_: bool
            Whether or not to print the result.

        Usage
        ------
        .. code-block:: python

            df = pd.DataFrame({'a': range(3),'b':range(1,4),'a_dup':range(3)})
            df.bp.get_duplicate_columns()
        """
        df = self._obj

        # make category columns as object
        for c in df.select_dtypes('category').columns.to_list():
	        df[c] = df[c].astype('object')
        groups = df.columns.to_series().groupby(df.dtypes).groups
        dupe_cols = []

        for _, v in groups.items():
            cols = df[v].columns
            ser = df[v]
            M = len(cols)
            for i in range(M):
                arr1 = ser.iloc[:,i].values
                for j in range(i+1, M):
                    arr2 = ser.iloc[:,j].values
                    if np.array_equal(arr1, arr2):
                        dupe_cols.append(cols[j])
                        if print_:
                            print(f'{cols[i]} == {cols[j]}')
                        break
        return dupe_cols

    def missing(self, thr=80,style=True,print_=False,
                plot_=True,odir='images',ofile=None,save=True,show=False,
                figsize=(8,6),fontsize=12,
                add_text=True,percent=True,comma=True,
                decimals=4,rot=75):
        """Get nice table of columns description of given dataframe.

        Parameters
        ----------
        df: pandas.DataFrame
            Input dataframe.
        thr: int
            Threshold for missing value percentage
        style: bool
            Whether or not to style the output dataframe.
        print_: bool
            Whether or not to print the output dataframe
        plot_: bool
            Whether or not to plot the output dataframe
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image. eg. missing_values.png
        save: bool
            Whether to save the image or not.
        show: bool
            Whether or not to show the plot.
        figsize: (int,int)
            Figure size
        fontsize: int
            Fontsize
        add_text: bool
            Whether or not to add text to bar plots.
        percent: bool
            Simply add % sign to text on bar plot.
        comma: bool
            Whether or not to style the number with comma in text in barplots.
        decimals: int
            Number of decimal points in bar plot text.
        rot: int
            Degree to rotate the text on barplot.

        Usage
        ------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.missing(thr=70)

        """
        df = self._obj
        df_desc = pd.DataFrame()
        df_desc['Feature'] = df.columns
        df_desc['Type'] = df.dtypes.values
        df_desc['Count'] = len(df)
        df_desc['Missing'] = df.isnull().sum().values
        df_desc['Zeros'] = df.eq(0).sum().values
        df_desc['Unique'] = df.nunique().values

        df_desc['MissingPct'] = df_desc['Missing'].div(len(df)).mul(100).values
        df_desc['ZerosPct'] = df_desc['Zeros'].div(len(df)).mul(100).values
        df_desc1 = df.describe().T
        df_desc = df_desc.merge(df_desc1,left_on='Feature',right_index=True,how='left')
        df_desc = df_desc.fillna('')

        df_missing = df_desc[df_desc['MissingPct'] > 0]
        df_missing = df_missing.sort_values(by=['MissingPct'],
                                                ascending=False)

        # columns
        cols_missing_high = df_missing.loc[df_missing['MissingPct']>=thr,
                                        'Feature'].to_list()
        cols_missing_low = [i for i in df_missing['Feature']
                            if i not in cols_missing_high]
        N = df_missing.shape[0]

        if plot_:
            ser = ser = df_missing.set_index('Feature')['MissingPct']\
                                .loc[lambda x: x>0]
            if len(ser) > 0:
                fig,ax = plt.subplots(figsize=figsize)
                ax = ser.plot.bar(color=sns.color_palette('Reds_r',N),
                                fontsize=12,ax=ax)
                if add_text:
                    add_text_barplot(ax, decimals=decimals,rot=rot,comma=comma,percent=percent)
                ax.set_ylim(0,100.00001)
                ax.set_ylabel('Percent',fontsize=fontsize)
                ax.axhline(y=80,ls='--',alpha=0.8,color='pink')
                ax.tick_params(axis='x', labelsize=fontsize)
                ax.tick_params(axis='y', labelsize=fontsize)
                ax.set_yticks(range(0,101,10))
                plt.title('Features with Missing Values',fontsize=fontsize+2)
                plt.tight_layout()

                if ofile:
                    # make sure this is base name
                    assert ofile == os.path.basename(ofile)
                    if not os.path.isdir(odir): os.makedirs(odir)
                    ofile = os.path.join(odir,ofile)
                else:
                    if not os.path.isdir(odir): os.makedirs(odir)
                    ofile = os.path.join(odir,f'missing_values.png')

                if save: plt.savefig(ofile, dpi=300)
                if show: plt.show(); plt.close()

        if style:
            df_missing_styled = (df_missing.style
                .apply(lambda x: ["background: salmon"
                if  v == 'object' else "" for v in x], axis = 1)
                ).background_gradient(subset=['MissingPct','ZerosPct'])

            df_missing = df_missing_styled

        if print_:
            print(df_missing)

        # print useful info
        print(f"Missing values high threshold = {thr}%")
        print(f"\nNumber of missing values features: {N}")
        print(f"cols_missing_high = {cols_missing_high}")
        print(f"cols_missing_low = {cols_missing_low}")

        return df_missing

#============== Plotting Numerical and Categorical Features =======
    def plot_num(self,col,xlim=None,figsize=(18,12),fontsize=24,xticks=None,
                ms=None,hist_kws={},bins=None,
                odir='images',ofile=None,save=True,show=False,
                print_=False,disp=False):
        """Plot numerical column.

        Parameters
        -----------
        col: str
            Name of the numerical column.
        xlim: list
            X-axis limit. eg. [0,100]
        figsize: (int,int)
            figure size. e.g. (12,8)
        fontsize: int
            fontsize
        xticks: list
            Xticks for plots.
        ms: int or string
            mpl style name. eg. ggplot, seaborn_darkgrid, 0-8
        hist_kws: dict
            distplot histogram kwargs
        bins: int
            Number of bins in histogram.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image. eg. col.png
        save: bool
            Whether to save the image or not.
        show: bool
            Whether or not to show the image.
        print_: bool
            Print output dataframe or not.
        disp: bool
            Display output dataframe or not.
        """
        df = self._obj
        mpl_style = get_mpl_style(ms)
        plt.style.use(mpl_style)
        if not is_numeric_dtype(df[col]):
            raise AttributeError(f'{col} must be a numeric column')

        if disp:
            display(df[col].describe().to_frame().T)

        if print_:
            print(df[col].describe().to_frame().T)

        x = df[col].dropna()

        if xlim:
            x = x[x>=xlim[0]]
            x = x[x<=xlim[1]]

        if not hist_kws:
            hist_kws={'histtype': 'bar','edgecolor':'black','alpha': 0.2}

        # NOTE: bins is parameter itself in distplot, not in kws.

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

        sns.distplot(x, ax=axes[0][0], hist_kws=hist_kws,
                        color='blue',bins=bins)
        sns.distplot(np.log(x[x>0]), ax=axes[0][1],
                        hist_kws=hist_kws, color='green',bins=bins)

        sns.boxplot(x, ax=axes[1][0],color='purple')
        sns.violinplot(x, ax=axes[1][1],color='y')

        # labels
        axes[0][0].set_xlabel(col,fontsize=fontsize)
        axes[0][1].set_xlabel(f'log({col}) (>0)',fontsize=fontsize)
        axes[1][0].set_xlabel(col,fontsize=fontsize)
        axes[1][1].set_xlabel(col,fontsize=fontsize)

        # xticks
        if xticks:
            axes[0][0].set_xticks(xticks)
            axes[1][0].set_xticks(xticks)
            axes[1][1].set_xticks(xticks)

        # title
        axes[0][0].set_title(f'Distribution of **{col}**',fontsize=fontsize)
        axes[0][1].set_title(f'Distribution of **log({col})**',fontsize=fontsize)

        # ticklabels
        for i,j in [[0,0],[0,1],[1,0],[1,1]]:
            axes[i][j].tick_params(axis='x', labelsize=fontsize)
            axes[i][j].tick_params(axis='y', labelsize=fontsize)

        # layout
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'{col}.png')

        if save: plt.savefig(ofile, dpi=300)
        if show: plt.show(); plt.close()

    def plot_cat(self,cat,figsize=(12,8),fontsize=14,odir='images',ms=None,
                ofile=None,save=True,show=False,print_=False,
                text_kw1={'comma': True},
                text_kw2={'percent': True}
                ):
        """Plot the categorical feature.

        Parameters
        -----------
        cat: str
            categorical feature to plot.
        figsize: (int,int)
            Figure size.
        fontsize: int
            Fontsize of labels and ticks.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ms: int or string
            mpl style name. eg. ggplot, seaborn_darkgrid, 0-8
        ofile: str
            Base name of output image. e.g cat.png
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the plot.
        print_: bool
            Whether or not to print category counts.
        text_kw1: dict
            Parameter dictionary for add_text.
        text_kw2: dict
            Parameter dictionary for add_text.

        """
        df = self._obj
        mpl_style = get_mpl_style(ms)
        plt.style.use(mpl_style)
        fig, axes = plt.subplots(1,2,figsize=figsize)
        df1 = df[cat].value_counts()

        df1_pct = df1.div(df1.sum()).mul(100)

        df1.plot.bar(color=sns.color_palette('magma',len(df1)),ax=axes[0])
        df1_pct.plot.bar(color=sns.color_palette('magma',len(df1)),ax=axes[1])

        add_text_barplot(axes[0],**text_kw1)
        add_text_barplot(axes[1],**text_kw2)

        axes[0].set_xlabel(cat,fontsize=fontsize)
        axes[1].set_xlabel(cat,fontsize=fontsize)
        axes[0].set_ylabel('Count',fontsize=fontsize)
        axes[1].set_ylabel('Percent',fontsize=fontsize)
        plt.subplots_adjust(top=0.72)
        plt.suptitle(f"Class distribution of {cat}",fontsize=fontsize+2)
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'{cat}.png')

        if save: plt.savefig(ofile, dpi=300)
        if show: plt.show(); plt.close()

        if print_:
            print('='*50)
            print(f'Feature: **{cat}**')
            print('Overall Count: ')
            for i,v in df1_pct.round(2).items():
                print(f'    {i}: {v}%')

    def plot_num_num(self,num1,num2,figsize=(12,8),fontsize=18,ms=None,
                    xticks1=None,xticks2=None,rot1=0,rot2=None,rot=None,
                    odir='images',
                    ofile=None,save=True,show=False):
        """Plot the numeric feature.

        Parameters
        -----------
        num1: str
            Numerical column 1
        num2: str
            Numerical column 2
        fontsize: int
            Fontsize of x and yticklabels
            (NOT xlabel and ylabels)
        xticks1: list
            xticks for first column.
        xticks2: list
            xticks for second column.
        rot1: int
            Rotation for xticklabels for column 1.
        rot2: int
            Rotation for xticklabels for column 1.
        rot: int
            Rotation for xticks of both columns. (overrides)
        ms: int or string
            mpl style name. eg. ggplot, seaborn_darkgrid, 0-8
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image. eg. num1_num2.png
        save: bool
            Whether to save the image or not.
        show: bool
            Whether or not to show the image.
        """
        df = self._obj
        mpl_style = get_mpl_style(ms)
        plt.style.use(mpl_style)

        if not is_numeric_dtype(df[num1]):
            raise AttributeError(f'{num1} must be a numeric column')
        if not is_numeric_dtype(df[num2]):
            raise AttributeError(f'{num2} must be a numeric column')

        hist_kws={'histtype': 'bar',
                    'edgecolor':'black',
                    'alpha': 0.2}

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

        sns.distplot(df[num1], ax=axes[0][0], hist_kws=hist_kws,color='blue')
        sns.distplot(df[num2], ax=axes[0][1], hist_kws=hist_kws,color='purple')
        sns.scatterplot(x=num1, y=num2, data=df,ax=axes[1][0],color='cadetblue')
        sns.scatterplot(x=num2, y=num1, data=df,ax=axes[1][1],color='olive')

        # title
        plt.suptitle(f'Scatterplot of {num1} vs {num2}',fontsize=fontsize,y=1.02)

        # labels
        axes[0][0].set_xlabel(num1,fontsize=fontsize)
        axes[0][1].set_xlabel(num2,fontsize=fontsize)
        axes[1][0].set_xlabel(num1,fontsize=fontsize)
        axes[1][1].set_xlabel(num2,fontsize=fontsize)

        axes[0][0].set_ylabel('kde',fontsize=fontsize)
        axes[0][1].set_ylabel('kde',fontsize=fontsize)
        axes[1][0].set_ylabel(num2,fontsize=fontsize)
        axes[1][1].set_ylabel(num1,fontsize=fontsize)

        # xticks1
        if xticks1:
            axes[0][0].set_xticks(xticks1)
            axes[1][0].set_xticks(xticks1)
            axes[1][1].set_yticks(xticks1)

        # xticks2
        if xticks2:
            axes[0][1].set_xticks(xticks2)
            axes[1][0].set_yticks(xticks2)
            axes[1][1].set_xticks(xticks2)

        # ticklabels fontsize
        for i,j in [[0,0],[0,1],[1,0],[1,1]]:
            axes[i][j].tick_params(axis='x', labelsize=fontsize)
            axes[i][j].tick_params(axis='y', labelsize=fontsize)

        # xticklabels1 rotation
        if rot2 == None: rot2 = rot1
        if rot != None: rot1 = rot; rot2 = rot

        plt.setp(axes[0][0].get_xticklabels(), ha="right", rotation=rot1)
        plt.setp(axes[1][0].get_xticklabels(), ha="right", rotation=rot1)

        # xticklabels2 rotation
        plt.setp(axes[0][1].get_xticklabels(), ha="right", rotation=rot2)
        plt.setp(axes[1][1].get_xticklabels(), ha="right", rotation=rot2)

        # layout
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'{num1}_{num2}.png')

        if save: plt.savefig(ofile, dpi=300)
        if show: plt.show(); plt.close()

    def plot_num_cat(self, num,cat,
                    figsize=(24,18),ms=None,
                    bins=100,fontsize=34,
                    odir='images',
                    ofile=None,save=True,show=False,
                    print_=False,disp=False):
        """Plot of continuous variable vs binary-target.

        Parameters
        ----------
        df: pandas.DataFrame
            Input data.
        num: str
            Numerical feature which is to be plotted.
        cat: str
            Categorical feature.
        figsize: (int, int)
            Figure size.
        ms: int or string
            mpl style name. eg. ggplot, seaborn_darkgrid, 0-8
        bins: int
            Number of bins in the histogram.
        fontsize: int
            Font size of xlabel and ylabels of all plots.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image. eg. mycol_target.png
        save: bool
            Whether to save the image or not.
        show: bool
            Whether or not to show the image.
        print_: bool
            Print output dataframe or not.
        disp: bool
            Display output dataframe or not.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.plot_num_cat('age','pclass')
        """
        df = self._obj
        mpl_style = get_mpl_style(ms)
        plt.style.use(mpl_style)
        if not is_numeric_dtype(df[num]):
            raise AttributeError(f'{num} must be a numeric column')

        df = df.dropna(subset=[num])
        unq = sorted(df[cat].dropna().unique())
        fig, ax = plt.subplots(2,2,figsize=figsize)

        # top-left violin and top-right boxplot
        sns.violinplot(x=cat,y=num,data=df,ax=ax[0][0])
        sns.boxplot(x=cat, y=num, data=df,ax=ax[0][1])

        # bottom left two-distplots
        for u in unq:
            ser = df.query(f" {cat} == @u")[num]
            sns.distplot(ser,bins=bins,label=f"{cat}_{u}", ax=ax[1][0])

        # bottom right stripplot
        sns.stripplot(x=cat,y=num,data=df,ax=ax[1][1])

        # label name
        ax[0][0].set_xlabel(cat,fontsize=fontsize)
        ax[0][0].set_ylabel(num,fontsize=fontsize)
        ax[0][1].set_xlabel(cat,fontsize=fontsize)
        ax[0][1].set_ylabel(num,fontsize=fontsize)
        ax[1][1].set_xlabel(cat,fontsize=fontsize)
        ax[1][1].set_ylabel(num,fontsize=fontsize)
        ax[1][0].set_xlabel(num,fontsize=fontsize)
        ax[1][0].set_ylabel(cat,fontsize=fontsize)

        # ticklabel
        ax[0][0].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[0][1].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[1][0].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[1][1].tick_params(axis='both', which='major', labelsize=fontsize)

        # show legend
        ax[1][0].legend(fontsize=fontsize)
        plt.suptitle(f"Plot of {num} vs {cat}",size=fontsize+2, y=1.12)
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'{num}_{cat}.png')

        if save: plt.savefig(ofile, dpi=300)
        if show: plt.show(); plt.close()

        # display
        if disp:
            display(df.groupby(cat)[num]
            .describe().round(2).T.add_prefix(f'{cat}_').T
            .append(df[num].describe().round(2)))

        if print_:
            print(df.groupby(cat)[num]
            .describe().round(2).T.add_prefix(f'{cat}_').T
            .append(df[num].describe().round(2)))

    def plot_cat_num(self,cat,num,figsize=(32,24),ms=None,
                        add_text=True,
                        comma=True,decimals=2,rot=75,
                        fontsize=48,odir='images',
                        ofile=None,save=True,show=False):
        """Plot the categorical feature against numerical feature.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        cat: str
            Categorical feature.
        num: str
            Numerical feature.
        figsize: (int,int)
            Figure size. e.g (12,8) gives nice figure for titanic.
        ms: int or string
            mpl style name. eg. ggplot, seaborn_darkgrid, 0-8
        add_text: bool
            Whether or not to add the text to plots.
        comma: bool
            Whether or not to style the number with comma in text in barplots.
        decimals: int
            Number of decimal points in bar plot text.
        rot: int
            Degree to rotate the text on barplot.
        fontsize: int
            Font size of xlabel and ylabels of all plots.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image. eg. mycol_target.png
        save: bool
            Whether to save the image or not.
        show: bool
            Whether or not to show the image.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.plot_cat_num('pclass','fare')
        """
        df = self._obj
        mpl_style = get_mpl_style(ms)
        plt.style.use(mpl_style)
        if not is_numeric_dtype(df[num]):
            raise AttributeError(f'{num} must be a numeric column')

        fig, ax = plt.subplots(3,2,figsize=figsize)

        df1 = df[cat].value_counts(normalize=True).reset_index()
        order = df1['index'].tolist()[::-1]

        pal = 'twilight'

        sns.stripplot(x=cat,y=num, data=df, ax=ax[0][0],order=order,palette=pal)
        sns.violinplot(x=cat,y=num, data=df, ax=ax[0][1],order=order,palette=pal)

        sns.barplot(x=cat,y=num, data=df, ax=ax[1][0],order=order,palette=pal)
        (df.groupby(cat)[num].mean() / df.groupby(cat)[num].mean().sum())\
        .plot.bar(color=sns.color_palette(pal,len(order)),ax=ax[1][1])

        sns.countplot(df[cat], order=order,palette=pal,ax=ax[2][0])
        sns.barplot(x='index',y=cat, data=df1,order=order,palette=pal,ax=ax[2][1])

        if add_text:
            for i,j in [[1,0],[1,1],[2,0],[2,1]]:
                add_text_barplot(ax[i][j],fontsize=fontsize,comma=comma,
                            decimals=decimals,rot=rot)

        # tick title
        ax[0][0].set_xlabel('',fontsize=fontsize)
        ax[0][1].set_xlabel('',fontsize=fontsize)
        ax[0][0].set_ylabel(num,fontsize=fontsize)
        ax[0][1].set_ylabel('',fontsize=fontsize)

        ax[1][0].set_xlabel('',fontsize=fontsize)
        ax[1][1].set_xlabel('',fontsize=fontsize)
        ax[1][0].set_ylabel(num,fontsize=fontsize)
        ax[1][1].set_ylabel('',fontsize=fontsize)

        ax[2][0].set_xlabel(cat,fontsize=fontsize)
        ax[2][1].set_xlabel(cat,fontsize=fontsize)
        ax[2][0].set_ylabel('count',fontsize=fontsize)
        ax[2][1].set_ylabel('',fontsize=fontsize)

        # ticklabel
        ax[0][0].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[0][1].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[1][0].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[1][1].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[2][0].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[2][1].tick_params(axis='both', which='major', labelsize=fontsize)

        # remove ticks
        ax[0][0].tick_params(axis='x', which='major', labelsize=0)
        ax[0][1].tick_params(axis='x', which='major', labelsize=0)
        ax[1][0].tick_params(axis='x', which='major', labelsize=0)
        ax[1][1].tick_params(axis='x', which='major', labelsize=0)

        # title
        plt.suptitle(f"Plot of {cat} vs {num}",size=fontsize+2, y=1.12)
        ax[0][0].set_title(f"Swarm plot of {cat} vs {num}",fontsize=fontsize+2)
        ax[0][1].set_title(f"Violin plot of {cat} vs {num}",fontsize=fontsize+2)
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'{cat}_{num}.png')

        if save:plt.savefig(ofile)
        if show: plt.show(); plt.close()

    def plot_cat_cat(self,cat,ycat,figsize=(12,12),ms=None,
                    ylim2=None,rot=80,fontsize=18,comma=True,
                    loc='upper left',
                    hide_xticks=False,odir='images',
                    ofile=None,save=True,show=False,print_=True):
        """Plot 3*2 plot for categorical feature vs target-categorical feature.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        cat: str
            categorical feature.
        ycat: str
            categorical feature for y axis.
        figsize: (int, int)
            Figure size.
        ms: int or string
            mpl style name. eg. ggplot, seaborn_darkgrid, 0-8
        ylim2: int
            Second plot y-limit upper range.
        rot: int
            Degree to rotate the text on barplot.
        fontsize: int
            Fontsize of xyticks and xylabels.
        comma: bool
            Whether or not to style the number with comma in text in barplots.
        loc: str
            matplotlib plot loc (location) variable. 0 1 2 3 or 'upper right' etc.
        hide_xticks: bool
            Hide xticks of top and middle rows.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image. eg. mycol_target.png
        save: bool
            Whether to save the image or not.
        show: bool
            Whether or not to show the image.
        print_: bool
            Whether or not to plot the category counts and percents.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.plot_cat_cat('pclass','survived')
        """
        df = self._obj
        mpl_style = get_mpl_style(ms)
        plt.style.use(mpl_style)

        # data
        rare = df[ycat].value_counts().idxmin()
        df1 = df.groupby([cat,ycat]).count().iloc[:,0].unstack().sort_values(rare)
        order = df1.index.tolist()

        # bottom right
        df1_pct = df1.sum(axis=1).div(df1.sum().sum()).mul(100).round(2)

        # mid right
        df1_pct_cat =  df1.div(df1.sum()).mul(100).round(2)
        df1_pct_cat_sorted = df1_pct_cat.sort_values(rare,ascending=False)
        df1_pct_cat_order = df1_pct_cat.loc[order]

        # top right
        # pd.crosstab(df['BMI_cat'],df['Response8'],normalize='index')
        df1_per_cat = df1.T.div(df1.sum(axis=1)).mul(100).T.round(2)
        df1_per_cat_order = df1_per_cat.loc[order]
        df1_per_cat_sorted = df1_per_cat.sort_values(rare,ascending=False)

        pal = 'magma'
        color=sns.color_palette(pal,df[cat].nunique()) # list of rgb
        ncats = df1.shape[1]
        blues = sns.color_palette("Blues",ncats)
        greens = sns.color_palette("Greens",ncats)

        fig, ax = plt.subplots(3,2,figsize=figsize)

        # top left :
        df1.plot.bar(ax=ax[0][0],color=blues)
        add_text_barplot(ax[0][0],rot=rot,comma=comma)
        ax[0][0].legend(loc=loc)

        # top right:
        df1_per_cat_order.plot.bar(ax=ax[0][1],color=greens)
        add_text_barplot(ax[0][1], percent=True,rot=rot)
        ax[0][1].legend(loc=loc)

        # mid left:  feature vs target count (same as )
        df1.plot.bar(ax=ax[1][0],color=blues)
        add_text_barplot(ax[1][0],rot=rot,comma=comma)
        ax[1][0].legend(loc=loc)

        # mid right percent plot
        df1_pct_cat_order.plot.bar(ax=ax[1][1],color=greens)
        add_text_barplot(ax[1][1], percent=True,rot=rot)
        ax[1][1].legend(loc=loc)

        # bottom left : count plot
        df1.sum(axis=1).rename(ycat).plot.bar(ax=ax[2][0],color=color)
        add_text_barplot(ax[2][0],rot=rot,comma=comma)

        # bottom right : percent plot
        df1_pct.rename(ycat).plot.bar(ax=ax[2][1],color=color)
        add_text_barplot(ax[2][1], percent=True,rot=rot)

        # limits
        if ylim2:
            ax[1][1].set_ylim(0,ylim2)

        # tick title
        ax[0][0].set_xlabel('',fontsize=fontsize)
        ax[0][1].set_xlabel('',fontsize=fontsize)
        ax[0][0].set_ylabel(ycat,fontsize=fontsize)
        ax[0][1].set_ylabel('',fontsize=fontsize)

        ax[1][0].set_xlabel('',fontsize=fontsize)
        ax[1][1].set_xlabel('',fontsize=fontsize)
        ax[1][0].set_ylabel(ycat,fontsize=fontsize)
        ax[1][1].set_ylabel('',fontsize=fontsize)

        ax[2][0].set_xlabel(cat,fontsize=fontsize)
        ax[2][1].set_xlabel(cat,fontsize=fontsize)
        ax[2][0].set_ylabel(ycat,fontsize=fontsize)
        ax[2][1].set_ylabel('',fontsize=fontsize)

        # ticklabel
        ax[0][0].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[0][1].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[1][0].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[1][1].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[2][0].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[2][1].tick_params(axis='both', which='major', labelsize=fontsize)

        # hide ticks
        if hide_xticks:
            ax[0][0].tick_params(axis='x', which='major', labelsize=0)
            ax[0][1].tick_params(axis='x', which='major', labelsize=0)
            ax[1][0].tick_params(axis='x', which='major', labelsize=0)
            ax[1][1].tick_params(axis='x', which='major', labelsize=0)

        ax[0][0].set_title(f"",fontsize=fontsize+2)
        ax[0][1].set_title(f"",fontsize=fontsize+2)

        # title
        plt.suptitle(f"Plot of {cat} vs {ycat}",size=fontsize+2, y=1.02)
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'{cat}_{ycat}.png')

        if save: plt.savefig(ofile)
        if show: plt.show()

        # print
        if print_:
            print('='*50)
            print(f'Feature: **{cat}**')
            print('Overall Count: ')
            for i,v in df[cat].value_counts(ascending=False,normalize=True
                    ).mul(100).round(2).items():
                print(f'    {i}: {v}%')

            print()
            print(f'Total  **{ycat}_{rare}** distribution:')
            for i,v in df1_pct_cat_sorted[rare].items():
                print(f'    {i}: {v}%')

            print()
            print(f'Per {cat}  **{ycat}_{rare}** distribution:')
            for i,v in df1_per_cat_sorted[rare].items():
                print(f'    {i}: {v}%')

    def plot_cat_cat2(self,cat,target_cat,
        figsize=(12,8),ylim2=None,odir='images',
        ofile=None,save=True,show=False):
        """Plot 2*2 plot for categorical feature vs target-cateogoical feature.

        Parameters
        ----------
        df: pandas.DataFrame
            Input data.
        cat: str
            Categorical feature.
        cat: str
            Categorical target feature.
        figsize: (int, int)
            Figure size.
        ylim2: int
            Upper limit of yaxis.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.plot_cat_cat('pclass','survived')

        """
        df = self._obj
        fig, ax = plt.subplots(2,2,figsize=figsize)

        # single count plot
        df[cat].value_counts(ascending=True).plot.bar(
        color=sns.color_palette('magma',df[cat].nunique()),ax=ax[0][0])

        # single percent plot
        df[cat].value_counts(ascending=True,normalize=True).mul(100).round(2)\
            .plot.bar(color=sns.color_palette('magma',df[cat].nunique()),
                        ax=ax[0][1])

        # double count  plot
        column0 = pd.crosstab(df[cat],df[target_cat]).columns.min()
        pd.crosstab(df[cat],df[target_cat]).sort_values(column0).plot.bar(ax=ax[1][0])

        # double percent plot
        pd.crosstab(df[cat],df[target_cat],normalize='index')\
        .sort_values(column0).mul(100).round(2).plot.bar(ax=ax[1][1])

        if ylim2:
            ax[1][1].set_ylim(0,ylim2)

        plt.suptitle(f'Count and Percent plot for {cat} vs {target_cat}',
                fontsize=14,color='blue')

        add_text_barplot(ax[0][0])
        add_text_barplot(ax[0][1], percent=True)
        add_text_barplot(ax[1][0])
        add_text_barplot(ax[1][1], percent=True)

        ax[0][0].legend()
        ax[0][1].legend()
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'{cat}_vs_{target_cat}.png')

        if save: plt.savefig(ofile)
        if show: plt.show()

    def plot_cat_stacked(self,cols,figsize=(12,8),fontsize=14,ms=None,
                odir='images',
                ofile=None,save=True,show=False,
                kws={}):
        """Plot stacked plot of categorical features.

        Parameters
        -----------
        cols: list
            List of the categorical column.
        figsize: (int,int)
            figure size. e.g. (12,8)
        fontsize: int
            fontsize
        ms: int or string
            mpl style name. eg. ggplot, seaborn_darkgrid, 0-8
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image. eg. col.png
        save: bool
            Whether to save the image or not.
        show: bool
            Whether or not to show the image.
        kws: dict
            Kwargs dict for pandas dataframe plot.
            e.g. kws = {'rot':90}
        """
        df = self._obj
        mpl_style = get_mpl_style(ms)
        plt.style.use(mpl_style)

        (df[cols_bin]
        .apply(lambda x: x.value_counts(normalize=True))
        .T
        .plot(kind='bar', stacked=True,figsize=figsize,fontsize=fontsize,**kws)
        )

        # layout
        plt.title('Proportion of zeros and ones in binary features',
                fontsize=fontsize,y=1.05)
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'cats_stacked_barchart.png')

        if save: plt.savefig(ofile, dpi=300)
        if show: plt.show(); plt.close()

#======================= Timeseries Analysis ==================
    def plot_date_cat(self,col_date,target_cat,figsize=(8,6),
                    odir='images',
                    ofile=None,save=True,show=False):
        """Seasonal plot of datetime column vs target cat.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        col_date: str
            datetime feature.
        target_cat: str
            binary target feature
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Usage:
        -------
        ts = pd.date_range(start='1/1/2018',
                            end='2/1/2019',freq='H')
        target = np.random.choice([0,1],size=len(ts))
        df_ts = pd.DataFrame({'date': ts, 'target': target})
        df_ts.bp.plot_date_cat('date','target',save=True,show=False)

        Note
        -------
        To see full images in jupyter notebook use this:
        .. code-block:: python

            %%javascript
            IPython.OutputArea.auto_scroll_threshold = 9999;

        """
        import calendar

        df = self._obj

        if not is_datetime64_any_dtype(df[col_date]):
            raise AttributeError(f'{col_date} must be a datetime column.')

        # day name
        map_dayofweek = dict(zip(range(7),list(calendar.day_name)))
        unq = df[col_date].dt.dayofweek.unique()
        order = pd.Series(unq).sort_values().map(map_dayofweek)
        x = df[col_date].dt.day_name()
        df.groupby([x,target_cat]).count().iloc[:,0].unstack()\
        .loc[order].plot(marker='o',figsize=figsize)
        plt.title(f'Seasonal variation of **{target_cat}** with **Day of Week**')
        plt.ylabel('Count')
        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f"{col_date}_vs_{target_cat}_dow_count.png")

        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show(); plt.close()

        # month name
        map_month = dict(zip(range(1,13),list(calendar.month_name)[1:]))
        unq = df[col_date].dt.month.unique()
        order = pd.Series(unq).sort_values().map(map_month)
        x = df[col_date].dt.month_name()
        df.groupby([x,target_cat]).count().iloc[:,0].unstack()\
            .loc[order].plot(marker='o',figsize=figsize)
        plt.title(f'Seasonal variation of **{target_cat}** wrt **Month**')
        plt.ylabel('Count')

        ofile = os.path.join(odir,f"{col_date}_vs_{target_cat}_monthly_count.png")
        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show(); plt.close()

        # day of month
        agg = 'day'
        x =  getattr(df[col_date].dt, agg)
        df.groupby([x,target_cat])\
            .count().iloc[:,0].unstack().plot(marker='o',figsize=figsize)
        plt.title(f'Seasonal variation of **{target_cat}** wrt **Day of Month**')
        plt.xticks(range(x.max()+1))
        plt.ylabel('Count')

        ofile = os.path.join(odir,f"{col_date}_vs_{target_cat}_daily_count.png")
        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show(); plt.close()

        # hour
        agg = 'hour'
        x = getattr(df[col_date].dt, agg)
        df.groupby([x,target_cat])\
            .count().iloc[:,0].unstack().plot(marker='o',figsize=figsize)
        plt.title(f'Seasonal variation of **{target_cat}** wrt **Hour of Day**')
        plt.xticks(range(x.max()+1))
        plt.ylabel('Count')

        ofile = os.path.join(odir,f"{col_date}_vs_{target_cat}_hourly_count.png")
        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show(); plt.close()

    def plot_daily_cat(self,col_date,target_cat,figsize=(12,8),
            show_xticks=True,odir='images',
            ofile=None,save=True,show=False):
        """Daily total plot of binary target.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        col_date: str
            Datetime feature.
        target_cat: str
            Categorical target variable.
        figsize: (int, int)
            Figure size.
        show_xticks: bool
            Whether or not to show the xticks.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Examples
        ---------
        .. code-block:: python

        ts_small = pd.date_range(start='1/1/2018',
                            end='2/1/2018',freq='H')
        target = np.random.choice([0,1],size=len(ts_small))
        df_ts_small = pd.DataFrame({'date': ts_small, 'target': target})

        df_ts_small.bp.plot_daily_cat('date','target')

        NOTE
        -----
        This function plots the daily sum for categorical target.
        Make sure data frame does not have too may days in x-axis, otherwise
        the plot looks ugly.

        """
        df = self._obj
        if not is_datetime64_any_dtype(df[col_date]):
            raise AttributeError(f'{col_date} must be a datetime column.')

        x = df[col_date].dt.date
        df.groupby([x,target_cat]).count().iloc[:,0].unstack()\
            .plot(marker='o',figsize=figsize)
        xtickvals = pd.date_range(x.min(),x.max(),freq='D')
        plt.xticks(xtickvals, rotation=90)
        plt.tight_layout()

        if not show_xticks:
            plt.xticks([])

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f"{col_date}_vs_{target_cat}_daily.png")

        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show(); plt.close()

#==================== Statistics ==================================
    def plot_corr(self,cols=None,target=None,topN=10,cmap='RdYlGn',
        annot=True,figsize=(12,8),annot_fontsize=12,
        xrot=0,yrot=0,fontsize=18,ytitle=1.05,mask=True,
        odir='images',ofile=None,save=True,show=False):
        """Correlation plot.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        cols: list,optional
            List of columns.
        target: str
            Name of target column
        topN: int
            Top N correlated features with target.
        cmap: str,optional
            Colormap. eg. RdBu, coolwarm,PuBu, RdYlGn
        annot: bool
            Show annotation or not.
        figsize: (int,int)
            Figure size.
        annot_fontsize: int
            Annotation fontsize.
        xrot: int
            Rotation of xtick labels
        yrot: int
            Rotation of ytick labels
        fontsize: int
            Size of x and y ticklabels
        ytitle: float
            Position of title above the figure.
        mask: bool
            Show only upper diagonal values.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of the output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.
        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.plot_corr()

            target = 'survived'
            df_few_cols = df[['age','fare']]
            df_few_cols.merge(df[target],left_index=True,right_index=True)\
                       .bp.plot_corr()

        """
        df = self._obj
        cols_orig = cols
        if not cols:
            cols = df.columns

        df_corr = df[cols].corr()
        if target:
            colsN = df_corr.nlargest(topN, target).index
            df_corr = df[colsN].corr()

        plt.figure(figsize=figsize)

        if mask:
            with sns.axes_style("white"):
                g = sns.heatmap(df_corr,vmin=-1,vmax=1,cmap=cmap,
                            mask=np.tril(df_corr),
                            annot_kws={'fontsize':annot_fontsize},annot=annot)

        else:
            g = sns.heatmap(df_corr,vmin=-1,vmax=1,cmap=cmap,
                            annot_kws={'fontsize':annot_fontsize},annot=annot)

        g.set_yticklabels(g.get_yticklabels(),
                        rotation=yrot, fontsize=fontsize)
        g.set_xticklabels(g.get_xticklabels(),
                        rotation=xrot, fontsize=fontsize)
        plt.title('Correlation Heatmap',fontsize=fontsize+4,y=ytitle)
        plt.tick_params(axis='both',which='major',labelsize=fontsize,
                labelbottom=False,bottom=False,top=False,labeltop=True)
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            name = '_'.join(cols)
            name = 'few_columns' if len(name) > 50 else name
            ofile = os.path.join(odir,f'corrplot_' + name + '.png')

        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show()

    def plot_corr_style(self,cols=None,target=None,topN=10,cmap='RdBu'):
        """Correlation plot with style and magnification.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        cols: list,optional
            List of columns.
        target: str
            Name of target column
        topN: int
            Top N correlated features with target.
        cmap: str,optional
            Colormap. eg. RdBu, coolwarm,PuBu

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.plot_corr_style()

        """
        df = self._obj
        if not cols:
            cols = df.columns

        df_corr = df[cols].corr()
        if target:
            colsN = df_corr.nlargest(topN, target).index
            df_corr = df[colsN].corr()

        out = df_corr.style.background_gradient(cmap, axis=1)\
        .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
        .set_caption("Hover to magify")\
        .set_precision(2)\
        .set_table_styles(magnify())

        return out

    def plot_corr_sns(self,cols=None,target=None,topN=10,fontsize=None,
                    xrot=90,yrot=0,odir='images',
                    ofile=None,save=True,show=True):
        """Correlation plot with Pearson correlation coefficient.
        Diagonals are distplots, right are scatterplots and left are kde.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        cols: list
            List of columns.
        target: str
            Name of target column
        topN: int
            Top N correlated features with target.
        fontsize: int
            Fontsize for xylabels and ticks
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.plot_corr_sns()

        """
        df = self._obj

        if not cols:
            cols = df.columns

        df_corr = df[cols].corr()
        cols = df_corr.columns
        if target:
            colsN = df_corr.nlargest(topN, target).index
            df_corr = df[colsN].corr()
            cols = df_corr.columns

        if not fontsize:
            if len(cols) <= 3:
                fontsize = 14
            else:
                fontsize = len(cols) * 5

        g = sns.PairGrid(data=df_corr, vars = cols, height = 3.5)

        g.map_upper(plt.scatter,color='#8A2BE2')
        g.map_diag(sns.distplot)
        g.map_lower(sns.kdeplot, cmap="Blues_d")
        g.map_lower(self._annotate_pearsonr)

        # label rotation
        for ax in g.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot,fontsize=fontsize)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=xrot,fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), rotation=xrot,fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), rotation=yrot, fontsize=fontsize,ha='right',va='center')

        # layout
        plt.suptitle('Correlation plot among features',fontsize=fontsize+4,y=1.05)
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            name = '_'.join(cols)
            name = 'few_columns' if len(name) > 50 else name
            ofile = os.path.join(odir,f'corr_scatterplot_' + name + '.png')

        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show(); plt.close()

    def corr_high(self,thr, print_=False,disp=True):
        """Get the most correlated features above given threshold.

        Parameters
        ----------
        df: pandas.DataFrame
            Input dataframe.
        thr: float
            Thereshold between 0 and 1.
        print_: bool
            Whether or not to print the output dataframe.

        Note:
        1. Only numerical features have correlation.
        2. Here we only get absolute correlation.

        Returns
        ----------
        df1 : pandas.DataFrame
            Output dataframe with most correlated features.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.corr().style.apply(lambda x: ["background: salmon" if  (abs(v) > 0.5 and v!=1) else "" for v in x], axis = 1)
            df.bp.corr_high(thr=0.5)

        """
        df = self._obj
        df_high_corr = (df.corr()
        .unstack()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'level_0':'feature1',
                            'level_1':'feature2',
                            0:'corr'})
        .query('feature1 != feature2')
        .assign(
        tmp = lambda dfx: dfx[['feature1', 'feature2']]\
                    .apply(lambda x: '_'.join(sorted(tuple(x))),
                        axis=1)
            )
        .drop_duplicates('tmp')
        .drop('tmp',axis=1)
        .query(' abs(corr) > @thr')
        .reset_index(drop=True)
        )

        # print useful info
        cols_high_corr = list(set(df_high_corr[['feature1',
                            'feature2']].to_numpy().ravel()))
        cols_high_corr1 = df_high_corr['feature1'].to_list()
        cols_high_corr2 = df_high_corr['feature2'].to_list()
        cols_high_corr_drop = np.setdiff1d(cols_high_corr1,cols_high_corr2).tolist()

        print(f'cols_high_corr = {cols_high_corr}')
        print(f'cols_high_corr1 = {cols_high_corr1}')
        print(f'cols_high_corr2 = {cols_high_corr2}')
        print(f'cols_high_corr_drop = {cols_high_corr_drop}')

        # print
        if print_:
            print(df_high_corr)

        # display
        if disp:
            return display(df_high_corr.style.background_gradient(subset=['corr']))

        return df_high_corr

    def corr_high_lst(self, thr,print_=False):
        """Get the most correlated features list above given threshold.

        Parameters
        ----------
        df: pandas.DataFrame
            Input dataframe.
        print_: bool
            Whether or not to print the output dataframe.
        thr: float
            Thereshold between 0 and 1.

        Returns
        ----------
        cols_corr : list
            List of most correlated features.

        Note
        ------
        1. Only numerical features have correlation.
        2. Here we only get absolute correlation.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.corr_high_lst(thr=0.5)
        """
        df = self._obj
        cols_high_corr_drop = set()
        df_corr = df.corr()
        for i in range(len(df_corr.columns)):
            for j in range(i):
                if (( abs(df_corr.iloc[i, j]) >= thr) and
                    (df_corr.columns[j] not in cols_high_corr_drop)):
                    colname = df_corr.columns[i]
                    cols_high_corr_drop.add(colname)
        if print_:
            print(df1)

        cols_high_corr_drop = list(cols_high_corr_drop)
        print(f'cols_high_corr_drop = {cols_high_corr_drop}')
        return cols_high_corr_drop

    def partial_corr(self,cols=None,print_=False,thr=1.0,disp=False):
        """Partial correlation coefficent among multiple columns of given array.

        Parameters
        ----------
        df: pandas.DataFrame
            Input dataframe.
        cols: list
            list of feature names
        print_: bool
            Whether or not to print the output dataframe.
        thr: float
            Threshold between 0 and 1.
        show: bool
            Whether or not to show the styled output dataframe.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.partial_corr()

        """
        import scipy

        df = self._obj
        if not cols:
            cols = df.select_dtypes('number').columns

        df = df[cols].dropna(how='any').copy()
        arr = df[cols].to_numpy()
        p = arr.shape[1]
        P_corr = np.zeros((p, p), dtype=np.float)
        for i in range(p):
            P_corr[i, i] = 1
            for j in range(i+1, p):
                idx = np.ones(p, dtype=np.bool)
                idx[i] = False
                idx[j] = False
                beta_i = scipy.linalg.lstsq(arr[:, idx], arr[:, j])[0]
                beta_j = scipy.linalg.lstsq(arr[:, idx], arr[:, i])[0]
                res_j = arr[:, j] - arr[:, idx].dot( beta_i)
                res_i = arr[:, i] - arr[:, idx].dot(beta_j)
                corr = scipy.stats.pearsonr(res_i, res_j)[0]
                P_corr[i, j] = corr
                P_corr[j, i] = corr
        df_partial_corr = pd.DataFrame(data=P_corr,columns=cols,index=cols)
        if print_:
            print(df_partial_corr)

        if disp:
            df1 = df_partial_corr.style.apply(lambda x: ["background: salmon"
                    if  (abs(v) > 0.5 and v!=1) else "" for v in x], axis = 1)
            return df1
        return df_partial_corr

    def outliers_tukey(self,num,thr=1.5,plot_=True,
            odir='images', ofile=None,save=True,show=False):
        """Get outliers bases on Tukeys Inter Quartile Range.

        Parameters
        -----------
        df: pandas dataframe
            Pandas dataframe.
        num: str
            Name of numerical column.
        thr: float
            Thereshold.
        plot_: bool
            Whether or not to plot.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Example
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            ser_outliers = df.bp.outliers_tukey('age')
        """
        df = self._obj

        if not is_numeric_dtype(df[num]):
            raise AttributeError(f'{num} must be a numeric column')

        # We need to drop nans
        ser = df[num].dropna()

        q1 = np.percentile(ser, 25)
        q3 = np.percentile(ser, 75)
        iqr = q3-q1
        low = q1 - thr*iqr
        high = q3 + thr*iqr
        idx_outliers = list(ser.index[(ser < low)|(ser > high)])
        ser_outliers = ser.loc[idx_outliers].to_frame()

        if plot_:
            sns.boxplot(x=ser)
            plt.title(f'Box plot of {num}')
            plt.tight_layout()

            if ofile:
                # make sure this is base name
                assert ofile == os.path.basename(ofile)
                if not os.path.isdir(odir): os.makedirs(odir)
                ofile = os.path.join(odir,ofile)
            else:
                if not os.path.isdir(odir): os.makedirs(odir)
                ofile = os.path.join(odir,f'{num}_boxplot.png')

            if save: plt.savefig(ofile,dpi=300)
            if show: plt.show(); plt.close()

        return ser_outliers

    def outliers_kde(self,num,thr=0.05):
        """Find outliers using KDEUnivariate method.

        Parameters
        -----------
        df: pandas dataframe
            Pandas dataframe.
        num: str
            Name of numerical column. The column must NOT have nans.
        thr: float
            Thereshold. Default is 5% (ie. 0.05)
        plot_: bool
            Whether or not to plot.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Example
        ---------
        .. code-block:: python
            df = sns.load_dataset('titanic')
            df1 = df.dropna(subset=['age']).reset_index(drop=True)
            idx_outliers, val_outliers = df1.bp.outliers_kde('age')
            df1.loc[idx_outliers,['age']]

        NOTE
        ----
        This method uses nonparametric way to estimate outliers.
        It captures the outliers even in cases of bimodal distributions.

        Ref: http://www.statsmodels.org/devel/generated/statsmodels.nonparametric.kde.KDEUnivariate.html

        """
        df = self._obj
        if not is_numeric_dtype(df[num]):
            raise AttributeError(f'{num} must be a numeric column')

        assert df[num].isna().sum() == 0, 'Missing values are not allowed.'

        x = df[num].to_numpy()

        from sklearn.preprocessing import scale
        from statsmodels.nonparametric.kde import KDEUnivariate

        x_scaled = scale(list(map(float, x)))
        kde = KDEUnivariate(x_scaled)
        kde.fit(bw="scott", fft=True)
        pred = kde.evaluate(x_scaled)

        n = sum(pred < thr)
        idx_outliers = np.asarray(pred).argsort()[:n]
        val_outliers = np.asarray(x)[idx_outliers]

        return idx_outliers, val_outliers

    def compare_kde_binn(self,cols,binn,m=1,n=1,bw=0.5,
                    figsize=(12,8),fontsize=14,loc='best',
                    odir='images',
                    ofile=None,save=True,show=False):
        """Compare the KDE plots of two numerical features against binary target.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        cols: list
            List of numerical columns.
        binn: str
            Binary target feature.
        m: int
            Number of plot rows
        n: int
            Number of plot columns.
        bw: float
            Bandwidth of kde plot.
        figsize: (int,int)
            Figure size.
        fontsize: int
            Size of x and y ticklabels.
        loc: str or int
            Location of legend. eg. 'best', 'lower left', 'upper left'
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.compare_kde(['age','fare'],'survived',1,2)

        """
        if isinstance(cols,str) or isinstance(cols,int):
            cols = [cols]
        df = self._obj[cols+[binn]].dropna(how='any')

        assert sorted(df[binn].unique().tolist()) == [0,1], 'Binary target must have values 0 and 1'

        t0 = df.loc[df[binn] == 0]
        t1 = df.loc[df[binn] == 1]

        fig, ax = plt.subplots(m,n,figsize=figsize)

        for i,col in enumerate(cols):
            plt.subplot(m,n,i+1)
            sns.kdeplot(t0[col], bw=bw,label=f"{binn} = 0")
            sns.kdeplot(t1[col], bw=bw,label=f"{binn} = 1")
            plt.ylabel('')
            plt.xlabel(col, fontsize=fontsize)
            locs, labels = plt.xticks()
            plt.tick_params(axis='both', which='major', labelsize=fontsize)
            plt.legend(prop=dict(size=fontsize),loc=loc)

        # remove empty subplots
        for i in range(m*n-len(cols)):
            ax.flat[-(i+1)].set_visible(False)

        # title
        plt.title(f'Compare kde plots')
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'compare_kde_cats_vs_{binn}.png')

        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show(); plt.close()

    def compare_kde2(self,num,binn,figsize=(12,8),fontsize=14,
                    odir='images',
                    ofile=None,save=True,show=False):
        """Compare the KDE plots of two numerical features against binary target.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        num: str
            Numerical feature.
        binn: str
            Binary feature.
        figsize: (int,int)
            Figure size.
        fontsize: int
            Size of x and y ticklabels.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.compare_kde('fare','survived')

        References
        -----------

        `stackoverflow <https://stackoverflow.com/questions/62375034/find-non-overlapping-area-between-two-kde-plots-in-python>`_
        """
        note = """I assume this is not working. When I compared it with sns.kde
                with binary target "Response8" and "risk_Age_medium_bool"
                variable of Prudential Life Insurance data, it gave me
                bimodal distribution, but clearly it is unimodel.
                So this function does not work properly.
        """
        if note:
            return "Currently not available."

        df = self._obj[[num,binn]].dropna(how='any')

        assert sorted(df[binn].unique().tolist()) == [0,1], 'Binary target must have values 0 and 1'

        x0 = df.loc[df[binn] == 0, num]
        x1 = df.loc[df[binn] == 1, num]

        kde0 = stats.gaussian_kde(x0, bw_method=0.3)
        kde1 = stats.gaussian_kde(x1, bw_method=0.3)

        xmin = min(x0.min(), x1.min())
        xmax = min(x0.max(), x1.max())
        dx = 0.2 * (xmax - xmin)    # add a 20% margin,
                                    # as the kde is wider than the data
        xmin -= dx
        xmax += dx

        x = np.linspace(xmin, xmax, 500)
        kde0_x = kde0(x)
        kde1_x = kde1(x)
        inters_x = np.minimum(kde0_x, kde1_x)

        plt.plot(x, kde0_x, color='b', label='No')
        plt.fill_between(x, kde0_x, 0, color='b', alpha=0.2)
        plt.plot(x, kde1_x, color='orange', label='Yes')
        plt.fill_between(x, kde1_x, 0, color='orange', alpha=0.2)
        plt.plot(x, inters_x, color='r')
        plt.fill_between(x, inters_x, 0, facecolor='none',
                        edgecolor='r', hatch='xx', label='intersection')

        area_inters_x = np.trapz(inters_x, x)

        handles, labels = plt.gca().get_legend_handles_labels()
        labels[2] += f': {area_inters_x * 100:.1f} %'
        plt.legend(handles, labels, title=binn)
        plt.title(f'{num} vs {binn}')
        plt.tight_layout()

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'compare_kde_{num}_vs_{binn}.png')

        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show(); plt.close()

#============ Miscellaneous Plots ===============================
    def countplot(self,cols,m,n,
                    figsize=(12,8),fontsize=14,rot=45,
                    odir='images',
                    ofile=None,save=True,show=False):
        """Multiple countplots in a grid of m*n subplots.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        cols: list
            List of numerical columns.
        binn: str
            Binary target feature.
        m: int
            Number of plot rows
        n: int
            Number of plot columns.
        figsize: (int,int)
            Figure size.
        fontsize: int
            Size of x and y ticklabels.
        rot: int
            Rotation of text on bar plots.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.countplot(['pclass','parch'],1,2)

        """
        df = self._obj[cols]
        fig, ax = plt.subplots(m,n,figsize=figsize)
        for i,col in enumerate(cols):
            plt.subplot(m,n,i+1)
            color=sns.color_palette('husl',df[col].nunique())

            axx = df[col].value_counts(normalize=True).mul(100).plot.bar(color=color)

            plt.xlabel(col,fontsize=fontsize)
            plt.ylim(0,100)
            plt.tick_params(axis='both', which='major', labelsize=fontsize)
            plt.tight_layout()

            for p in axx.patches:
                txt = str(p.get_height().round(2)) + '%'
                txt_x = p.get_x()
                txt_y = p.get_height()
                plt.text(txt_x,txt_y,txt,fontsize=fontsize-2,rotation=rot)

        for i in range(m*n-len(cols)):
            ax.flat[-(i+1)].set_visible(False)

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'countplots_for_cats.png')

        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show(); plt.close()

    def regplot_binn(self,cols1,cols2,binn,m,n,
            figsize=(12,8),fontsize=18,debug=False,
            odir='images',ofile=None,save=True,show=False):
        """Plot multiple lmplots in a grid of m*n subplot.

        Useful to analysize high correlated features.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        cols1: list
            List of numerical columns.
        cols2: list
            List of numerical columns.
        binn: str
            Binary target feature.
        m: int
            Number of plot rows
        n: int
            Number of plot columns.
        figsize: (int,int)
            Figure size.
        fontsize: int
            Size of x and y ticklabels.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.regplot_binn(['age'], ['fare'],1,2)
            sns.lmplot(x='age',y='fare',data=df,hue='survived')

        """
        df = self._obj
        assert sorted(df[binn].unique().tolist()) == [0,1], 'Binary target must have values 0 and 1'

        msg = f"""\
        sns.lmplot(x='{cols1[0]}',y='{cols2[0]}',data=df,hue='{binn}')
        """
        if debug:
            print(msg)

        fig, axes = plt.subplots(m,n,figsize=figsize)
        for i in range(len(cols1)):
            ax = plt.subplot(m,n,i+1)

            x0 = df.loc[df[binn]==0, cols1[i]]
            y0 = df.loc[df[binn]==0, cols2[i]]
            x1 = df.loc[df[binn]==1, cols1[i]]
            y1 = df.loc[df[binn]==1, cols2[i]]

            sns.regplot(x=x0, y=y0, ax=ax,label='0')
            sns.regplot(x=x1, y=y1, ax=ax,label='1')
            ax.legend(prop={'size':fontsize-2})

            plt.xlabel(cols1[i],fontsize=fontsize)
            plt.ylabel(cols2[i],fontsize=fontsize)
            plt.tick_params(axis='both', which='major', labelsize=fontsize)
            plt.tight_layout()

        plt.suptitle("Regression plots of Numerical features against Binary target.",
                    fontsize=fontsize+4,y=1.05)

        for i in range(m*n-len(cols1)):
            axes.flat[-(i+1)].set_visible(False)

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'regplots_of_nums_vs_{binn}.png')

        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show(); plt.close()

    def plot_boxplot_cats_num(self,cats,num,figsize=(12,8),
                odir='images',
                ofile=None,save=True,show=False):
        """Plot boxplots in a loop.

        Parameters
        ----------
        df: pandas.DataFrame
            Input data.
        cats: list
            List of categorical columns to plot boxplots.
        num: str
            Name of numerical target column.
        figsize: (int,int)
            Figure size.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.plot_boxplot_cats_num(['pclass','sex'],'age',show=True)

        """
        df = self._obj
        if not os.path.isdir(odir): os.makedirs(odir)
        for cat in cats:
            plt.figure(figsize=figsize)
            plt.title(f'Box plot of {cat} vs {num}')
            sns.boxplot(x=num, y=cat, data=df,width = 0.8,
                        orient = 'h', showmeans = True, fliersize = 3)

            ofile = os.path.join(odir,f"{cat}_vs_{num}_boxplot.png")
            if save: plt.savefig(ofile,dpi=300)
            if show: plt.show(); plt.close()

    def plot_count_cat(self, cat, percent=True,bottom=0,
                figsize=(12,8),fontsizex=18, fontsizey=18,
                horizontal=False,number=False,
                odir='images',
                ofile=None,save=True,show=False):
        """count plot of given column with optional percent display.

        Useful when just wanting to plot the counts. e.g while doing ML.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        cat: str
            Name of categorical column.
        percent: bool
            Whether or not the percent to be shown.
        bottom: float
            Adjust the matplotlib figure bottom.
        figsize: (int, int)
            Figure size.
        fontsizex: int
            Font size for xticks.
        fontsizey: int
            Font size for yticks.
        horizontal: bool
            Whether or not plot horizontal bar plot.
        number: int
            Whether or not to show to number on top of barplots.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Examples
        ---------
        .. code-block:: python

            df = sns.load_dataset('titanic')
            df.bp.plot_count_cat(['pclass',show=True)

        """
        df = self._obj
        plt.figure(figsize=figsize)
        total = float(len(df[cat]) )

        if not horizontal:
            ax = sns.countplot(x=cat, data=df, order=df[cat].value_counts().index)
            for p in ax.patches:
                xp = p.get_x()+p.get_width()/2.
                height = p.get_height()
                yp = height
                txt = '{:1.2f}%'.format((height/total)*100)

                if percent:
                    txt = '{:d} ({:1.2f}%)'.format(height, (height/total)*100)
                ax.text(xp,bottom,txt, ha="center", fontsize=24, rotation=90,color='black')

        if horizontal:
            ax = sns.countplot(y=cat, data=df, order=df[cat].value_counts().index)
            for p in ax.patches:
                xp = 0.1 # p.get_width() gives 0
                yp = p.get_y()

                # text
                txt_number = str(round(p.get_width(), 2))
                txt_pct = str(round(p.get_width()/total*100, 2)) + '%'

                if number:
                    percent = False
                    ax.text(xp,yp,txt_number, ha="left", va='top',rotation=0,color='black')

                if percent:
                    ax.text(xp,yp,txt_pct, ha="left", va='top',rotation=0,color='black')

        plt.xticks(fontsize=fontsizex,rotation=90)
        plt.yticks(fontsize=fontsizey)
        plt.xlabel(cat, fontsize=fontsizex)
        plt.ylabel('count', fontsize=fontsizey)
        plt.tight_layout()
        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'countplot_{cat}.png')

        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show(); plt.close()

    def plot_pareto(self,cat,value=None,thr=None,
                    figsize=(12,8),
                    rot=90,fontsize=18,
                    offset=0, decimals=2,
                    odir='images', ofile=None,
                    save=True,show=False):
        """Pareto Chart.

        Each category must be unique.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        cat: str
            Categorical column to get value counts and then plot.
        value: str
            Value column for categorical column of
            already aggregated dataframe.
            If we use grpby all the cat variables must be unique.
        thr: int
            Upper threshold for rare categories. e.g 98
        figsize: (int,int)
            Figure size.
        rot: int
            Rotation of text on bar chart.
        yrot: int
            Rotation of ytick labels
        fontsize: int
            Size of x and y ticklabels
        offset: int
            Offset of text above bar chart. eg. 1
        decmals: float
            Number of decimals to show in text in barchart.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Whether or not to save the image.
        show: bool
            Whether or not to show the image.

        Examples
        ---------
        .. code-block:: python
            tips = sns.load_dataset('tips')
            tips.bp.plot_pareto('size',thr=90)

            df = pd.DataFrame({'fruit': ['apple','banana','grape','plum'],
                    'price': [17,4,7,7]})
            df.bp.plot_pareto(cat='fruit',value='price',thr=80)

        """
        df_ = self._obj
        ylabel = 'Count'
        title = f'Pareto Chart for {cat}'

        if value:
            # Here cat = fruit value = price
            ylabel = value
            title = f'Pareto Chart for {cat}' + f' vs {value}'
            df = df_[[cat,value]].sort_values(value,ascending=False)
            df.index = df[cat].astype(str)
            df["cumperc"] = df[value].cumsum()/df[value].sum()*100

        else:
            # Here, cat = small categorical number or cat
            df = df_[cat].value_counts().to_frame()
            df.index = [str(i) for i in df.index]
            df["cumperc"] = df[cat].cumsum()/df[cat].sum()*100

        df_below = df

        # threshold
        if thr:
            df_below = df[df['cumperc'] < thr].copy()
            cumperc_above = df.loc[df['cumperc'] >= thr, cat].sum()
            if value:
                value_above = df.loc[df['cumperc'] >= thr, value].sum()
                df_below.loc['OTHERS'] = [0] * 3
                df_below.loc['OTHERS',cat] = 0
                df_below.loc['OTHERS',value] = value_above
                df_below.loc['OTHERS','cumperc'] = 100

            else:
                df_below.loc['OTHERS'] = [cumperc_above, 100]

        # plot
        fig, ax1 = plt.subplots(figsize=figsize)

        # barplot
        color = '#C71585'
        if value:
            ax1.bar(df_below.index, df_below[value], color=color)
        else:
            ax1.bar(df_below.index, df_below[cat], color=color)

        # line plot with twinx
        ax2 = ax1.twinx()
        ax2.plot(df_below.index, df_below["cumperc"], color="C1", marker="D", ms=7)

        # 100 % line
        ax2.axhline(y=100,color='r',lw=1,linestyle='--')

        # percent format
        ax2.yaxis.set_major_formatter(PercentFormatter())

        # tick colors
        ax1.tick_params(axis="y", colors=color)
        ax2.tick_params(axis="y", colors="C1")

        # ylim
        ax2.set_ylim([0,135])

        # add text to barplot
        if value:
            for p, (idx,row) in zip(ax1.patches,df_below[[value,'cumperc']].iterrows()) :
                x,y = p.get_x(), p.get_height()
                val,pareto = row[value], row['cumperc']
                pareto = np.round(pareto, decimals=decimals)
                ax1.text(x,y+offset,f'{val:,.0f} ({pareto}%)',
                        fontsize=fontsize,color='blue',
                        rotation=rot,
                        ha='left')
        else:
            for p, (idx,row) in zip(ax1.patches,df_below[[cat,'cumperc']].iterrows()) :
                x,y = p.get_x(), p.get_height()
                val,pareto = row[cat], row['cumperc']
                pareto = np.round(pareto, decimals=decimals)
                ax1.text(x,y+offset,f'{val:,.0f} ({pareto}%)',
                        fontsize=fontsize,color='blue',
                        rotation=rot,
                        ha='left')

        # x and y label
        ax1.set_xlabel(cat,fontsize=fontsize+2)
        ax1.set_ylabel(ylabel,fontsize=fontsize+2)
        ax2.set_ylabel('Pareto',fontsize=fontsize+2)

        # tick fontsize
        ax1.xaxis.set_tick_params(labelsize=fontsize)
        ax1.yaxis.set_tick_params(labelsize=fontsize)
        ax2.yaxis.set_tick_params(labelsize=fontsize)

        # grid
        plt.grid('x')
        plt.tight_layout()
        plt.suptitle(title,fontsize=fontsize)
        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'pareto_chart_{cat}.png')

        if save: plt.savefig(ofile,dpi=300)
        if show: plt.show(); plt.close()

#======================= Plotly Plots ==============================
    def plotly_countplot(self,col,topN=None,color=None,
                    odir='images', ofile=None,save=True,
                    show=True,auto_open=False):
        """Value counts plot using plotly and pandas.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        col: str
            The variable name.
        topN: int
            Top n correlated variables to show in heatmap.
        color: str
            Color of count plot.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Save the html or not.
        show: bool
            Whether or not to show the rendered html in notebook.
        auto_open: bool
            Whether or not to automatically open the ouput html file.

        Example:
        ----------
        tips = sns.load_dataset('tips')
        tips.bp.plotly_countplot('day')
        """
        df = self._obj
        if not color:
            color='rgb(158,202,225)'

        df1 = df[col].value_counts()
        if not topN:
            topN = len(df1)
        df1 = df1.head(topN)

        trace0 = go.Bar(
                    x=df1.index.values,
                    y=df1.values.ravel(),
                    text=df1.values.tolist(),
                    textposition = 'auto',
                    marker=dict(
                        color=color,
                        line=dict(
                            color=color,
                            width=1.5),
                    ),
                    opacity=1.0
                )

        data = [trace0]
        layout = go.Layout(title='Count plot of ' + col,
                    xaxis=dict(title=col,tickvals= df1.index.values),
                    yaxis=dict(title='Count'))

        fig = go.Figure(data=data, layout=layout)

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'{col}.html')

        if save:
            plot(fig, filename=ofile,auto_open=auto_open)

        if show:
            return iplot(fig)

    def plotly_corr(self,target,topN=10,method='pearson',
            colorscale='Reds',width=800,height=800, ytitle=0.99,
            odir='images',ofile=None,save=True,show=True,auto_open=False):
        """Plot correlation heatmap for top N numerical columns.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        target: str
            Target variable name w.r.t which we choose top N other features.
            For example price. Then we get top N feaures most correlated
            with price.
        topN: int
            Top n correlated variables to show in heatmap.
        method: str
            Method of correlation. Default is 'pearson'
        colorscale: str
            Color scale of heatmap. Default is 'Reds'
        width: int
            Width of heatmap.
        height: int
            Height of heatmap.
        ytitle: float
            Position of title.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Save the html or not.
        show: bool
            Whether or not to show the rendered html in notebook.
        auto_open: bool
            Whether or not to automatically open the ouput html file.

        Examples
        ---------
        .. code-block:: python
            diamonds = sns.load_dataset('diamonds')
            diamonds.bp.plotly_corr('price',topN=4)

        """
        df = self._obj
        df_corr = df.corr(method=method)

        colsN = df_corr.nlargest(topN, target).index
        df_corr = df[colsN].corr()

        z = df_corr.values
        z = np.tril(z)
        annotation_text = np.array(
            ['{:.2f}'.format(i) for i in z.ravel()]).reshape(z.shape)

        fig = ff.create_annotated_heatmap(z,showscale=True,
                    colorscale=colorscale,
                    annotation_text=annotation_text,
                    x=df_corr.columns.values.tolist(),
                    y=df_corr.columns.values.tolist()
                    )
        fig['layout'].update(width=width,height=height)
        fig.update_layout(
            title = {
                'text': f'Correlation Plot of Top {topN} features with target **{target}**',
                'y': ytitle
            }
        )

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,f'correlation_plot.html')

        if save:
            plot(fig, filename=ofile,auto_open=auto_open)

        if show:
            return iplot(fig)

    def plotly_boxplot(self,cols=None,show_all_points=False,
                        width=600,height=800,odir='images',
                        ofile=None,show=True,save=True,auto_open=False):
        """Plot correlation heatmap for top N numerical columns.

        Parameters
        -----------
        df: pandas.DataFrame
            Input data.
        cols: list
            List of numerical features.
        show_all_points: bool
            Whether or not to show all the points in boxplot.
        width: int
            Width of heatmap.
        height: int
            Height of heatmap.
        odir: str
            Name of output directory.
            This directory will be created if it does not exist.
        ofile: str
            Base name of output image.
        save: bool
            Save the html or not.
        show: bool
            Whether or not to show the rendered html in notebook.
        auto_open: bool
            Whether or not to automatically open the ouput html file.

        Examples
        ---------
        .. code-block:: python
            titanic = sns.load_dataset('titanic')
            titanic.bp.plotly_boxplot('age')

        Note
        -----
        To display large ouput in jupyter notebook without scrolling use this:
        .. code-block:: python
            %%javascript
            IPython.OutputArea.auto_scroll_threshold = 9999
        """
        df = self._obj

        # select only first 10 numerical features if cols is none.
        if not cols:
            cols = df.select_dtypes('number').columns[:10]
            height = 300 * len(cols)
            width = 800

        if isinstance(cols,str) or isinstance(cols,int):
            cols = [cols]
            num = cols[0]
            if not is_numeric_dtype(df[num]):
                raise AttributeError(f'{num} must be a numeric column')

            ser = df[num].dropna()
            thr = 1.5

            q1 = np.percentile(ser, 25)
            q3 = np.percentile(ser, 75)
            iqr = q3-q1
            floor = q1 - thr*iqr
            ceiling = q3 + thr*iqr
            idx_outliers = list(ser.index[(ser < floor)|(ser > ceiling)])
            ser_outliers = ser.loc[idx_outliers].to_frame()

        traces = []
        for col in cols:
            trace = go.Box(
                y = df[col].dropna(),
                name = f"{col}",
                boxpoints = 'suspectedoutliers',
                marker = dict(
                    color = 'rgb(8,81,156)',
                    outliercolor = 'rgba(219, 64, 82, 0.6)',
                    line = dict(
                        outliercolor = 'rgba(219, 64, 82, 0.6)',
                        outlierwidth = 2)),
                line = dict(
                    color = 'rgb(8,81,156)')
            )
            traces.append(trace)

        fig = make_subplots(rows=len(cols), cols=1)
        for i in range(len(cols)):
            fig.add_trace(traces[i],row=i+1,col=1)

        # figure layout
        title = 'Boxplot'
        fig['layout'].update(width=width,height=height,title=title,title_x=0.5)

        if ofile:
            # make sure this is base name
            assert ofile == os.path.basename(ofile)
            if not os.path.isdir(odir): os.makedirs(odir)
            ofile = os.path.join(odir,ofile)
        else:
            if not os.path.isdir(odir): os.makedirs(odir)
            name = '_'.join(cols)
            name = 'few_columns' if len(name) > 50 else name
            ofile = os.path.join(odir,f'boxplot_' + name + '.html')

        if save:
            plot(fig, filename=ofile,auto_open=auto_open)

        if show:
            return display(iplot(fig))

        if isinstance(cols,str) or isinstance(cols,int):
            return ser_outliers

    def plotly_yearplot(self,val,date=None,index=False,cmap='Greens',text='text',year=None):
        """
        Plot a timeseries as a yearly heatmap which resembles like github
        contribution plot.

        Parameters
        ----------
        df : pandas dataframe
            Dataframe must have at least two columns `date` and `value`
        val : string
            Name of value column.
        date : string
            Name of date column.

        cmap : string, optional
            Colormap. Default is 'Greens'
        year : int, optional
            Year is required if data is more than one year.
        text : string, optional
            Text to display in hoverinfo

        Returns
        -------
        fig : plotly figure

        Examples
        --------

        .. plot::
            :context: close-figs

            df = pd.DataFrame({'date': pd.date_range('2020-02-01',
                                        '2020-12-31',freq='D')})
            df['value'] = np.random.randint(1,20,size=(len(df)))
            df.bp.plotly_yearplot('value','date')

        """
        df = self._obj
        colorscale = get_plotly_colorscale('Greens',df[val])

        # date is none
        if date == None:
            if "date" in df.columns:
                date = "date"

        # check if index is date
        if date not in df.columns:
            if type(df.index) == pd.core.indexes.datetimes.DatetimeIndex:
                date = df.index.name
                df = df.reset_index()
            else:
                raise "Please pass `date` parameter. Or make sure index is of type datetime "

        # copy data to avoid creating unwanted columns
        df = df[[date,val]].copy()

        if not text: text = 'text'
        if text not in df.columns:
            df[text] = (
                'Date: ' + df[date].dt.strftime("%Y %b %d %a") + ' ' +
                '(Value='+ df[val].astype(str) + ')'
                )

        # make sure data is single year
        if (df[date].min().year != df[date].max().year) and (year==None):
            raise """Please specify year if the data spans over multiple years.
                    e.g year=2020"""

        if (df[date].min().year != df[date].max().year) and (year!=None):
            df = df[df[date].dt.year==year]

        # plotly data
        data = [
        go.Heatmap(
                x = df[date].dt.weekofyear,
                y = df[date].dt.day_name(),
                z = df[val],
                text= df[text],
                hoverinfo=text,
                xgap=3, # this
                ygap=3, # and this is used to make the grid-like apperance
                showscale=False,
                colorscale=colorscale
                )
        ]

        layout = go.Layout(
            title='',
            height=280,
            yaxis=dict(
                showline = False, showgrid = False, zeroline = False,
                tickmode='array',
                tickvals=[0,1,2,3,4,5,6],
                ticktext=df[date].head(7).dt.strftime("%a"),
            ),
            xaxis=dict(
                showline=False, showgrid=False, zeroline=False,dtick=4

            ),
            font={'size':10, 'color':'#9e9e9e'},
            plot_bgcolor=('#fff'),
            margin = dict(t=40),
        )

        fig = go.Figure(data=data, layout=layout)
        return fig

#================================ Style ==================================
    def style_rowcol(self,name=0,axis=1,color=None,c=None,
                ):
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
        df = self._obj
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

    def style_diag(self,diag='both',c1='lightgreen',c2='salmon',):
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
        df = self._obj

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

    def style_cellv(self,cond='',c='lightgreen',idx=None,col=None):
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
            cond = "type(v) in [int,float] and 1<=v<=3"
            cond = "v in [3,'S']"
            df1.bp.style_cellv(cond)
            df1.bp.style_cellv("v==1",idx=2)
            df1.bp.style_cellv("v==1",col='pclass')

        """
        df = self._obj

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

    def style_cellx(self,cond,c='lightgreen'):
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
        df = self._obj

        def f(x,cond,c='lightgreen'):
            attr = f'background-color: {c}'
            df1 = x.astype(str).replace(x, '')
            df1 = df1.astype(str).replace('nan','')
            df1[eval(cond)] = attr
            return df1
        return df.style.apply(f,axis=None,cond=cond,c=c)
