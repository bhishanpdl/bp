__doc__ = """
Plotting functions for data science.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_distribution(df, col, ax=None, **kwargs):
    """
    Plot a histogram of the distribution of a given column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    col : str
        Column to plot.
    ax : matplotlib.Axes, optional
        Axes to plot on.
    kwargs : dict, optional
        Keyword arguments to pass to the matplotlib hist function.

    Returns
    -------
    ax : matplotlib.Axes
        Axes containing the plot.

    """
    if ax is None:
        ax = plt.gca()
    ax.hist(df[col], **kwargs)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    return ax