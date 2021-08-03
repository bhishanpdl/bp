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
all_ds_corr_outliers = [
    'corrwith',
    'corr_high',
    'corr_high_lst',
    'partial_corr',
    'point_biserial_correlation',
    'outliers_tukey',
    'outliers_kde'
    ]

all_ds_ds = [
    'freq_count',
    'get_column_descriptions',
    'report_cat_binn',
    'compare_kde_binn',
    'compare_kde2'
    ]

all_ds_json = [
    'parse_json_col',
    'MyJson'
    ]

all_ds_speed = ['optimize_memory']

all_hlp = ["hlp"]

all_ml_data_proc = [
    "get_outliers",
    "get_outliers_iqr",
    "get_outliers_tukey",
    "get_outliers_kde",
    "remove_outliers",
    "remove_outliers_iqr",
    "remove_outliers_tukey",
    "add_interactions",
    "select_kbest_features"
    ]

all_ml_modelling = [
    "print_df_eval",
    "adjustedR2",
    "multiple_linear_regression",
    "get_binary_classification_scalar_metrics",
    "get_binary_classification_scalar_metrics2",
    "get_binary_classification_report",
    "print_confusion_matrix",
    "get_false_negative_frauds",
    "plot_confusion_matrix_plotly",
    "plot_roc_auc",
    "plot_roc_skf"
    ]

all_ml_statsmodels = [
    "regression_residual_plots",
    "print_statsmodels_summary",
    "lm_stats",
    "lm_plot",
    "lm_residual_corr_plot"
    ]

all_pandas_api =  ["BPAccessor"]

all_plot_calendar = [
    'print_calendar_month',
    'display_calendar_month',
    'display_calendar_month_red_green',
    'display_calendar_month_cmap'
    ]

all_plot_corr = [
    "plot_corr",
    "plot_corr_style",
    "plot_corr_sns",
    "plot_corrplot_with_pearsonr",
    "plot_multiple_jointplots_with_pearsonr"
    ]

all_plot_ds = [
    "plot_stem",
    "plot_pareto",
    "countplot",
    "regplot_binn",
    "plot_two_clusters",
    ]

all_plot_map = [
    "plotly_usa_map",
    "plotly_usa_map2",
    "plotly_usa_map_agg",
    "plotly_usa_map_bubble",
    "plotly_country_map",
    "plotly_country_map_agg",
    "plotly_mapbox"
    ]

all_plot_modelling = [
    "plot_simple_linear_regression",
    "plotly_binary_clf_evaluation"
    ]

all_plot_num_cat = [
    "plot_num",
    "plot_cat",
    "plot_num_num",
    "plot_num_cat",
    "plot_cat_num",
    "plot_cat_cat",
    "plot_cat_stacked",
    "plot_boxplot_cats_num",
    "plot_count_cat",
    "plot_cat_cat2",
    "plot_num_cat2",
    "plot_cat_binn",
    "plot_cat_cat_pct",
    "plot_donut_binn",
    ]

all_plot_plotly =  [
    "Plotly_Charts",
    "plotly_corr",
    "plotly_corr_heatmap",
    "plotly_countplot",
    "plotly_histogram",
    "plotly_distplot",
    "plotly_radar_plot",
    "plotly_boxplot",
    "plotly_boxplot_allpoints_with_outliers",
    "plotly_boxplot_categorical_column",
    "plotly_scattergl_plot",
    "plotly_scattergl_plot_colorcol",
    "plotly_scattergl_plot_subplots",
    "plotly_bubbleplot",
    "plotly_mapbox"
    ]

all_plot_stats = [
    "plot_statistics",
    "plot_ecdf",
    "plot_gini",
    "plot_ks"
    "get_yprobs_sorted_proportions",
    ]

all_plot_tsa = [
    "plot_date_cat",
    "plot_daily_cat"
    ]

all_util_colors = [
    "rgb2hex",
    "hex_to_rgb",
    "get_distinct_colors",
    "discrete_cmap",
    "get_colornames_from_cmap"
    ]

all_pd_styles = [
    "highlight_row",
    "highlight_col",
    "highlight_diag",
    "highlight_rowf",
    "highlight_rowsf",
    "highlight_colf",
    "highlight_diagf",
    "highlight_rcd"
]

all_util_pd = [
    "describe",
    "make_date",
    "add_datepart",
    "add_elapsed_times",
    "cont_cat_split",
    "df_shrink_dtypes",
    "df_shrink",
]

all_util_plots = [
    "add_text_barplot",
    "light_axis",
    "no_axis",
    "magnify",
    "get_mpl_style",
    "get_plotly_colorscale"
    ]

all_utils = [
    'show_methods',
    'parallelize_dataframe',
    'ifnone'
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
