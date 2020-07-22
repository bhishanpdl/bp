__author__ = 'Bhishan Poudel'

__doc__ = """
This module helps fitting various machine leaning models.

- plot_simple_linear_regression(X_test, y_test, model,xlabel,ylabel,
    figsize=(6,5), data_color='salmon', pred_color='seagreen')

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.plot_simple_linear_regression(X_test, y_test, model,xlabel,ylabel)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

def plot_simple_linear_regression(X_test, y_test, model,xlabel,ylabel,
    figsize=(6,5), data_color='salmon', pred_color='seagreen'):
    plt.figure(figsize=figsize)

    plt.scatter(X_test, y_test, color=data_color, label="Data", alpha=.1)
    plt.plot(X_test, model.predict(X_test),color=pred_color,
            label="Predicted Regression Line")

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend()

    plt.show()