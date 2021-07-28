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
all_ds_ds = ['print_df_eval',
            'freq_count',
            'get_column_descriptions',
            'adjustedR2',
            'multiple_linear_regression',
            'get_high_correlated_features_df']

all_ds_json = ['parse_json_col']

all_ds_speed = ['optimize_memory']

all_ds_stats = ["partial_corr",
                "point_biserial_correlation",
                "find_corr"]

all_hlp = ["hlp"]

all_ml_data_proc = ["get_outliers_tukey",
        "get_outliers_kde",
        "remove_outliers_iqr",
        "add_interactions",
        "select_kbest_features"
        ]

all_ml_model_eval = ["get_binary_classification_scalar_metrics",
        "get_binary_classification_scalar_metrics2",
        "get_binary_classification_report",
        "print_confusion_matrix",
        "get_false_negative_frauds",
        "plot_confusion_matrix_plotly",
        "plot_roc_auc",
        "plot_roc_skf"
        ]

all_ml_statsmodels = ["regression_residual_plots",
        "print_statsmodels_summary",
        "lm_stats",
        "lm_plot",
        "lm_residual_corr_plot"]

all_pandas_api =  ["BPAccessor"]

all_plot_colors = ["rgb2hex",
                "hex_to_rgb",
                "get_distinct_colors",
                "discrete_cmap"]

all_plot_ds = ["plot_num",
        "plot_cat",
        "plot_num_num",
        "plot_num_cat",
        "plot_cat_num",
        "plot_cat_cat",
        "plot_date_cat",
        "plot_daily_cat",
        "plot_boxplot_cats_num",
        "plot_multiple_jointplots_with_pearsonr",
        "plot_corrplot_with_pearsonr",
        "plot_count_cat",
        "plot_corrplot_with_pearsonr",
        "plot_corr",
        "plot_corr_style",
        "plot_cat_cat2",
        "plot_num_cat2",
        "plot_pareto",
        "plot_cat_cat_pct",
        ]

all_plot_map = ["plotly_usa_map",
        "plotly_usa_map2",
        "plotly_agg_usa_plot",
        "plotly_usa_bubble_map",
        "plotly_country_plot",
        "plotly_agg_country_plot",
        "plotly_mapbox"
        ]

all_plot_model_eval = ["plotly_binary_clf_evaluation"]

all_plot_model = ["plot_simple_linear_regression"]

all_plot_plotly =  ["plotly_corr_heatmap",
                "plotly_countplot",
                "plotly_histogram",
                "plotly_distplot",
                "plotly_boxplot",
                "plotly_boxplot_allpoints_with_outliers",
                "plotly_boxplot_categorical_column",
                "plotly_scattergl_plot",
                "plotly_scattergl_plot_colorcol",
                "plotly_scattergl_plot_subplots",
                "plotly_bubbleplot",
                "plotly_mapbox"
                ]

all_plot_stats = ["get_yprobs_sorted_proportions",
                "plot_gini",
                "plot_ks"
                ]

all_plot_utils = ["add_text_barplot",
                "light_axis",
                "no_axis",
                "magnify",
                "get_mpl_style",
                "get_plotly_colorscale"
                ]

all_utils = ["show_methods",
            "parallelize_dataframe"
            ]

__all__ = ( all_ds_ds
            + all_ds_json
            + all_ds_speed
            + all_ds_stats
            + all_hlp
            + all_ml_data_proc
            + all_ml_model_eval
            + all_ml_statsmodels
            + all_pandas_api
            + all_plot_colors
            + all_plot_ds
            + all_plot_map
            + all_plot_model_eval
            + all_plot_model
            + all_plot_plotly
            + all_plot_stats
            + all_plot_utils
            + all_utils
            )

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Note: use __all__ variable on top of modules to determine what to
# display when we use import *


# data science
from bhishan.ds_ds import (print_df_eval,
                        freq_count,
                        get_column_descriptions,
                        adjustedR2,
                        multiple_linear_regression,
                        get_high_correlated_features_df)


# json
from bhishan.ds_json import parse_json_col

# data science speed
from bhishan.ds_speed import optimize_memory

# statistics
from bhishan.ds_stats import (partial_corr,
                            point_biserial_correlation,
                            find_corr)

# help
from bhishan.hlp import hlp

# data processing
from bhishan.ml_data_proc import (get_outliers_tukey,
    get_outliers_kde,
    remove_outliers_iqr,
    add_interactions,
    select_kbest_features
    )

# model evaluation
from bhishan.ml_model_eval import (get_binary_classification_scalar_metrics,
        get_binary_classification_scalar_metrics2,
        get_binary_classification_report,
        print_confusion_matrix,
        get_false_negative_frauds,
        plot_confusion_matrix_plotly,
        plot_roc_auc,
        plot_roc_skf
        )

from bhishan.ml_statsmodels import (regression_residual_plots,
        print_statsmodels_summary,
        lm_stats,
        lm_plot,
        lm_residual_corr_plot)

from bhishan.plot_colors import  (rgb2hex,
                                hex_to_rgb,
                                get_distinct_colors,
                                discrete_cmap)

from bhishan.plot_ds import (plot_num,
        plot_cat,
        plot_num_num,
        plot_num_cat,
        plot_cat_num,
        plot_cat_cat,
        plot_date_cat,
        plot_daily_cat,
        plot_boxplot_cats_num,
        plot_multiple_jointplots_with_pearsonr,
        plot_corrplot_with_pearsonr,
        plot_count_cat,
        plot_corrplot_with_pearsonr,
        plot_corr,
        plot_corr_style,
        plot_cat_cat2,
        plot_num_cat2,
        plot_pareto,
        plot_cat_cat_pct,
)

from bhishan.plot_map import (plotly_usa_map,
        plotly_usa_map2,
        plotly_agg_usa_plot,
        plotly_usa_bubble_map,
        plotly_country_plot,
        plotly_agg_country_plot,
        plotly_mapbox
)

from bhishan.plot_model_eval import plotly_binary_clf_evaluation

from bhishan.plot_model import plot_simple_linear_regression

from bhishan.plot_plotly import (plotly_corr_heatmap,
            plotly_countplot,
            plotly_histogram,
            plotly_distplot,
            plotly_boxplot,
            plotly_boxplot_allpoints_with_outliers,
            plotly_boxplot_categorical_column,
            plotly_scattergl_plot,
            plotly_scattergl_plot_colorcol,
            plotly_scattergl_plot_subplots,
            plotly_bubbleplot,
            plotly_mapbox
)

from bhishan.plot_stats import (get_yprobs_sorted_proportions,
                                plot_gini,
                                plot_ks
)


from bhishan.plot_utils import (add_text_barplot,
                light_axis,
                no_axis,
                magnify,
                get_mpl_style,
                get_plotly_colorscale
)

from bhishan.utils import show_methods, parallelize_dataframe








