__author__ = 'Bhishan Poudel'

__doc__ = """
This module helps fitting various machine leaning models.

Plotly Functions:
1.  plotly_corr_heatmap
2.  plotly_countplot
3.  plotly_distplot
4.  plotly_radar_plot
4.  plotly_boxplot
5.  plotly_boxplot_allpoints_with_outliers
6.  plotly_boxplot_categorical_column
7.  plotly_scattergl_plot
8.  plotly_scattergl_plot_colorcol
9.  plotly_scattergl_plot_subplots
10. plotly_bubbleplot
11. get_mapbox_access_token
12. plotly_mapbox
13.
14.
15.

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.plot_plotly?

"""
__all__ = [
    "Plotly_Charts",
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

# Imports
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
from pandas.core.frame import DataFrame, Series
from mytyping import (IN, SN, SI, SIN, TL, LD, DS, DSt, NUM, NUMN,
                        AD, AS, DN, ARR, ARRN, SARR, LIMIT, LIMITN)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
# from plotly.tools import make_subplots
from plotly.subplots import make_subplots
import plotly.tools as tls
from plotly.offline import plot, iplot

# I need these functions
# from plot_colors import get_distinct_colors
# from plot_colors import rgb2hex
# from plot_colors import hex_to_rgb

# Global parameters
COLORS30 = ['#696969', '#d3d3d3', '#8b4513',
                '#228b22', '#808000', '#483d8b',
                '#008b8b', '#4682b4', '#000080',
                '#8fbc8f', '#8b008b', '#b03060',
                '#ff4500', '#ff8c00', '#00ff00',
                '#9400d3', '#00ff7f', '#dc143c',
                '#00ffff', '#f4a460', '#0000ff',
                '#f08080', '#adff2f', '#1e90ff',
                '#f0e68c', '#ffff54', '#dda0dd',
                '#90ee90', '#ff1493', '#7b68ee']


class Plotly_Charts:
    """
    Interactive data visualization using plotly.
        - Scatter Plot
        - Bar Plot
        - Violin Plot
        - Box Plot
        - Distribution Plot
        - Histogram
        - Correlation Plot

    Args:
        df (pd.DataFrame): Input pandas dataframe having target column
        target (str): Target name
        exclude (str,list, optional): Columns to exclude. e.g. ID
        max_num (int, optional): Maximum number of rows to include.

    Returns:
        Charts widget

    Usage:
    df = sns.load_dataset('titanic')
    Chart = Plotly_Charts(df, exclude=["pclass", "age"], target="survived")
    charts = Chart.make_charts()
    """

    def __init__(self:Any,
        df:DataFrame,
        target:str,
        exclude:ARRN=None,
        max_num:IN=None
        ):
        """Init parameters.

        Args:
            df (pd.DataFrame): Input pandas dataframe having target column
            target (str): Target name
            exclude (str,list, optional): Columns to exclude. e.g. ID. Defaults to None.
            max_num (int, optional): Maximum number of rows to include. Defaults to None.
        """
        self.df = df
        self.target = target
        self.exclude = exclude
        self.max_num = max_num
        self.dfX = self.df.drop(self.target, axis=1).head(self.max_num)

        if self.exclude is not None:
            self.dfX = self.dfX.drop(self.exclude, axis=1)

    def make_charts(self):
        import pandas as pd

        import ipywidgets as widgets
        from ipywidgets import HBox, VBox, Button
        from ipywidgets import interact_manual, interactive

        import plotly.express as px
        import plotly.figure_factory as ff
        import plotly.offline as pyo
        import plotly.graph_objects as go
        from plotly.offline import iplot

        from IPython.display import display

        intr_kwargs = {"manual": True, "manual_name": "Make_Chart"}

        def display_html(txt:str):
            html_ = widgets.HTML(value=f"<h1> {txt} </h1>")
            display(html_)

        def display_four_boxes(plot_fn:Callable):
            widgets.interact_manual.opts["manual_name"] = "Make_Chart"
            hbox1 = interactive(plot_fn, intr_kwargs)
            hbox2 = interactive(plot_fn, intr_kwargs)
            hbox3 = interactive(plot_fn, intr_kwargs)
            hbox4 = interactive(plot_fn, intr_kwargs)

            g = widgets.HBox([hbox1, hbox2])
            display(g)
            h = widgets.HBox([hbox3, hbox4])
            display(h)

        display_html("Interactive Data Visualization using Plotly.")

        dfX = self.dfX
        dfX_num = dfX.select_dtypes("number")
        dfX_cat = dfX.select_dtypes(exclude="number")
        cols_num = list(dfX_num.columns) + [None]
        cols_cat = list(dfX_cat.columns) + [None]

        intr_kwargs = {"manual": True, "manual_name": "Make_Chart"}

        outs = [widgets.Output() for i in range(7)]

        tab = widgets.Tab(children=outs)
        tab.set_title(0, "Scatter Plot")
        tab.set_title(1, "Bar Plot")
        tab.set_title(2, "Violin Plot")
        tab.set_title(3, "Box Plot")
        tab.set_title(4, "Distribution Plot")
        tab.set_title(5, "Histogram")
        tab.set_title(6, "Correlation plot")

        display(tab)

        with outs[0]:
            display_html("Scatter Plots")
            x = widgets.Dropdown(options=cols_num)
            def scatter_plot(X_Axis=cols_num, Y_Axis=cols_num[1:],
                            Color=cols_num):
                marker_color = dfX[Color] if Color is not None else None
                fig = go.FigureWidget(
                    data=go.Scatter(
                        x=dfX[X_Axis],
                        y=dfX[Y_Axis],
                        mode="markers",
                        text=list(dfX_cat),
                        marker_color=marker_color,
                    )
                )

                fig.update_layout(
                    title=f"{Y_Axis.title()} vs {X_Axis.title()}",
                    xaxis_title=f"{X_Axis.title()}",
                    yaxis_title=f"{Y_Axis.title()}",
                    autosize=False,
                    width=600,
                    height=600,
                )
                fig['layout']['title']['x'] = 0.5
                fig.show()

            display_four_boxes(scatter_plot)

        with outs[1]:
            display_html("Bar Plots")

            def bar_plot(X_Axis=cols_cat, Y_Axis=cols_num, Color=cols_cat):
                color = dfX[Color] if Color is not None else None
                fig = px.bar(dfX, x=dfX[X_Axis], y=dfX[Y_Axis], color=color)
                fig.update_layout(
                    barmode="group",
                    title=f"{X_Axis.title()} vs {Y_Axis.title()}",
                    xaxis_title=f"{X_Axis.title()}",
                    yaxis_title=f"{Y_Axis.title()}",
                    autosize=False,
                    width=600,
                    height=600,
                )
                fig['layout']['title']['x'] = 0.5
                fig.show()

            display_four_boxes(bar_plot)

        with outs[2]:
            display_html("<h1>Violin Plots")

            def violin_plot(X_Axis=cols_num, Y_Axis=cols_num, Color=cols_cat):
                fig = px.violin(
                    dfX, X_Axis, Y_Axis, Color, box=True, hover_data=dfX.columns
                )
                fig.update_layout(
                    title=f"{X_Axis.title()} vs {Y_Axis.title()}",
                    xaxis_title=f"{X_Axis.title()}",
                    autosize=False,
                    width=600,
                    height=600,
                )
                fig['layout']['title']['x'] = 0.5
                fig.show()

            display_four_boxes(violin_plot)

        with outs[3]:
            display_html("<h1>Box Plots")

            def box_plot(X_Axis=cols_cat, Y_Axis=cols_num, Color=cols_cat):
                xaxis_title=f"{X_Axis.title()}" if not X_Axis is None else ''
                yaxis_title=f"{Y_Axis.title()}" if not Y_Axis is None else ''
                title=f"{xaxis_title} vs {yaxis_title}" if ( X_Axis and Y_Axis) else (X_Axis or Y_Axis).title()

                fig = px.box(dfX, x=X_Axis, y=Y_Axis, color=Color, points="all")

                fig.update_layout(
                    barmode="group",
                    title=title,
                    xaxis_title=xaxis_title,
                    yaxis_title=yaxis_title,
                    autosize=False,
                    width=600,
                    height=600,
                )
                fig['layout']['title']['x'] = 0.5
                fig.show()

            display_four_boxes(box_plot)

        with outs[4]:
            display_html("<h1>Distribution Plots")
            def dist_plot(X_Axis=cols_num, Y_Axis=cols_num, Color=cols_cat):
                xaxis_title=f"{X_Axis.title()}" if not X_Axis is None else ''
                yaxis_title=f"{Y_Axis.title()}" if not Y_Axis is None else ''
                title=f"{xaxis_title} vs {yaxis_title}" if ( X_Axis and Y_Axis) else (X_Axis or Y_Axis).title()

                fig = px.histogram(
                    dfX,
                    X_Axis,
                    Y_Axis,
                    Color,
                    marginal="violin",
                    hover_data=dfX.columns,
                )
                fig.update_layout(
                    title=title,
                    xaxis_title=xaxis_title,
                    autosize=False,
                    width=600,
                    height=600,
                )
                fig['layout']['title']['x'] = 0.5
                fig.show()

            display_four_boxes(dist_plot)

        with outs[5]:
            display_html("<h1>Histogram")

            def hist_plot(X_Axis=list(dfX.columns)):
                fig = px.histogram(dfX, X_Axis)
                fig.update_layout(
                    title=f"{X_Axis.title()}",
                    xaxis_title=f"{X_Axis.title()}",
                    autosize=False,
                    width=600,
                    height=600,
                )
                fig['layout']['title']['x'] = 0.5
                fig.show()

            display_four_boxes(hist_plot)

        with outs[6]:

            display_html("<h1>Correlation Plots")

            import plotly.figure_factory as ff

            df_corr = dfX.corr()
            colorscale = [
                "Greys","Greens","Bluered","RdBu","Reds","Blues",
                "Picnic","Rainbow","Portland","Jet","Hot","Earth",
                "Blackbody","Electric","Viridis","Cividis",
            ]

            @interact_manual
            def plot_corrs(colorscale=colorscale):
                fig = ff.create_annotated_heatmap(
                    z=df_corr.round(2).values,
                    x=list(df_corr.columns),
                    y=list(df_corr.index),
                    colorscale=colorscale,
                    annotation_text=df_corr.round(2).values,
                )
                fig['layout']['title']['x'] = 0.5
                iplot(fig)

def plotly_corr_heatmap(
    df:DataFrame,
    target:str,
    topN:int=10,
    method:str='pearson',
    colorscale:str='Reds',
    width:int=800,
    height:int=800,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False
    )->Any:
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
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.

    Examples
    ---------
    .. code-block:: python
        df = sns.load_dataset('diamonds')
        plotly_corr_heatmap(df,'price',topN=4)

    """
    df_corr = df.corr(method=method)
    cols10 = df_corr.nlargest(topN, target).index

    df_corr = df[cols10].corr()
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
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

def plotly_countplot(
    df:DataFrame,
    col:str,
    topN:IN=None,
    color:SN=None,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False
    ):
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
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.
    """
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
    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

def plotly_histogram(
    df:DataFrame,
    col:str,
    nbins:IN=None,
    size:IN=None,
    color:SN=None,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False
    ):
    """Histogram plot of given column of pandas dataframe using plotly.

    Parameters
    -------------
    df : pandas.DataFrame
        Pandas dataframe
    col : str
        Name of column
    nbins : int
        Number of bins to use in histogram.
    size : int
        Step size of x-axis bins.
    color : str
        Color of the plot. Valid matplotlib color name.

    """
    x = df[col].values

    if not color:
        color = 'rgb(102, 0, 102)'
    trace0 = go.Histogram(x=x,marker=dict(color=color))
    if nbins:
        trace0 = go.Histogram(x=x,
                            nbinsx=nbins,
                            marker=dict(color=color))

    if size:
        trace0 = go.Histogram(x=x,
                    marker=dict(color=color),
                    xbins=dict(start=np.min(x),size=1,end=np.max(x)),
                    )
    data = [trace0]
    layout = go.Layout(title='Histogram of ' + col,
                    xaxis=dict(title=col.title()),
                    yaxis=dict(title='Bin Counts'))
    fig = go.Figure(data=data, layout=layout)
    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

def plotly_distplot(
    df:DataFrame,
    cols:ARR,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False
    ):
    """Distribution plot using plotly.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list
        List of columns.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.

    """
    cols_list = [cols] if isinstance(cols, str) else cols
    df1 = df[cols_list]

    fig = ff.create_distplot([df1[c] for c in cols_list],
                            cols_list,
                            bin_size=.25,
                            show_rug=False,
                            curve_type='normal')
    head = 'Distribution Plot of '
    title = head + cols if isinstance(cols,str) else head + ', '.join(cols)
    fig['layout'].update(title=title)
    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

def plotly_radar_plot(
    df:DataFrame,
    target:str,
    categories:ARRN=None,
    names:ARRN=None,
    colors:ARRN=None,
    opacities:ARRN=[0.5,0.5],
    show_data:bool=False,
    show_obs:bool=False,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False):
    """Plot the Radar Chart or Spider Diagram or Polygon Plot for Binary Case.

    NOTE:
    1. The names and colors are picked based on sorted values of target.
    2. We use following formula to calculate values with y== 0 (or first unq)

    ZERO = (( df0 - df.min() )  / df_max_min).mean(axis=0)

    and not:
    ZERO = (( df0 - df0.min() )  / df_max_min).mean(axis=0)

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    target: str
        Name of target column.
    categories: list
        Column names.
    names: list
        Names of binary features. e.g. ['Benign','Malignant']
    colors: list
        List of colors. e.g. ['green','red']
    opacities: list
        List of opacities. e.g. [0.5,0.5]
    show_data: bool
        Whether or not to print the data
    show_obs: bool
        Whether or not to print example observations.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.

    """
    # unique values in target
    df = df.copy()
    df = df.sort_values(target)

    u = df[target].sort_values().unique()
    assert len(u) == 2, "The target must have 2 unique values. Look the the input dataframe."

    out = f"""
Observations:

The highest values for almost all parameters are obtained by -- case.
Feature5 have similar normalized mean between {names[0]} and {names[1]} cases.
    """
    if show_obs:
        print(out)

    bool_0 = df[target]==u[0]
    bool_1 = df[target]==u[1]

    df = df.drop(target,axis=1) # keep only features and exclude target

    if categories is None:
        categories = df.columns.tolist()

    if names is None:
        names = [str(i) for i in u]

    if colors is None:
        colors = ['blue', 'red']

    # min max scaling
    df_max_min = df.max(axis=0) - df.min(axis=0)

    df0 = df[bool_0]
    df1 = df[bool_1]

    ZERO = (( df0 - df.min() ) / df_max_min).mean(axis=0)
    ONE  = (( df1 - df.min() ) / df_max_min).mean(axis=0)

    if show_data:
        print('zero:\n', ZERO)
        print('\n\none:\n', ONE)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=ZERO,
        theta=categories,
        fill='toself',
        name=names[0],
        fillcolor=colors[0],
        opacity=opacities[0],
    ))

    fig.add_trace(go.Scatterpolar(
        r=ONE,
        theta=categories,
        fill='toself',
        name=names[1],
        fillcolor=colors[1],
        opacity=opacities[1],
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True
        )

    title = 'Radar Plot'
    fig['layout'].update(title=title)
    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

def plotly_boxplot(
    df:DataFrame,
    cols:ARRN,
    ylim_lst:ARR=None,
    color:SN=None,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False
    ):
    """Box plot using plotly.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list
        List of columns.
    ylim_lst: list
        List for yaxis limit.
    color: str
        Name of color for the boxplot.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.

    """
    cols_list = [cols] if isinstance(cols, str) else cols

    data = [go.Box(y=df[col],
                name=str(col),
                marker=dict(color=color),
                hoverinfo="name+y")
            for i,col in enumerate(cols_list) ]

    title = 'Boxplot'
    layout = go.Layout(title = title, yaxis=dict(range=ylim_lst))

    fig = go.Figure(data=data,layout=layout)
    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

def plotly_boxplot_allpoints_with_outliers(
    df:DataFrame,
    col:str,
    color:SN=None,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False
    ):
    """Box plot with all points and outliers using plotly.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col: str
        Name of column.
    color: str
        Name of color for the boxplot.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.

    """
    if not color:
        color = 'rgb(7,40,89)'
    trace0 = go.Box(
        y = df[col],
        name = "Box Plot with All Points",
        jitter = 0.3,
        pointpos = -1.8,
        boxpoints = 'all',
        marker = dict(color = color),
        line = dict(color = color)
        )

    trace1 = go.Box(y = df[col],
                    name = "Box Plot with Suspected Outliers",
                    boxpoints = 'suspectedoutliers')

    data = [trace0,trace1]

    layout = go.Layout(
        title = "Box Plot with Suspected Outliers for " + col
    )

    fig = go.Figure(data=data,layout=layout)
    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

def plotly_boxplot_categorical_column(
    df:DataFrame,
    xcol:str,
    ycol:str,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False):
    """Boxplot of categorical columns.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    xcol: str
        Name of x column.
    ycol: str
        Name of y column.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.

    """
    df[xcol] = df[xcol].astype(str)
    traces = []
    for unq in df[xcol].unique():
        if str(unq) == 'nan':
            traces.append(plotly.graph_objs.Box(y=df.loc[pd.isnull(df[xcol]), ycol],
                                                name = str(unq)
                                                )
                        )
        else:
            traces.append(plotly.graph_objs.Box(y=df.loc[df[xcol] == unq, ycol],
                                                name = unq
                                                )
                        )
    data = traces
    fig = go.Figure(data=data, layout=None)
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

def plotly_cat_binn_zero_one(
    df:DataFrame,
    cat:str,
    binn:str,
    zero:IS,
    one:IS,
    name:str,
    is_one_good:bool=False,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False) :
    """Plot categorical feature vs binary feature.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cat: str
        Name of categorical feature.
    binn: str
        Name of binary feature.
    zero: str,int
        Value of zero. e.g. 0, 'No'
    one: str,int
        Value of one. e.g. 1, 'Yes'
    name: str
        Name of binary feature. e.g. 'Churn', 'Fraud','Survived'
    is_one_good: bool
        Whether 1 is good or not.
        e.g 1 is good when Alive but not when Fraud.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.

    Examples
    ---------
    .. code-block:: python

        module_path = "/Users/poudel/Dropbox/a00_Bhishan_Modules/bhishan/"
        sys.path.append(module_path)

        import seaborn as sns
        from plot_plotly import plotly_cat_binn_zero_one

        df = sns.load_dataset('titanic')
        plotly_cat_binn_zero_one(df,'pclass','survived',0,1,
                                name='Alive',is_one_good=False)

    """
    # params
    color0 = 'green' if is_one_good else 'tomato'
    color1 = 'tomato' if is_one_good else 'green'
    df_zero = df[(df[binn] == zero)]
    df_one = df[(df[binn] == one)]
    dfx = pd.DataFrame(pd.crosstab(df[cat],df[binn]) )
    dfx['Attr%'] = dfx[one] / (dfx[one] + dfx[zero]) * 100
    dfx = dfx.sort_values(one, ascending = False)

    trace1 = go.Bar(
        x=df_zero[cat].value_counts().keys(),
        y=df_zero[cat].value_counts().values,
        text=df_zero[cat].value_counts().values,
        textposition = 'auto',
        name=f' No {name}', opacity = 0.8,
        marker=dict(color=color0,line=dict(color='#000000',width=1))
    )

    trace2 = go.Bar(
        x=df_one[cat].value_counts().keys(),
        y=df_one[cat].value_counts().values,
        text=df_one[cat].value_counts().values,
        textposition = 'auto',
        name=f'{name}',opacity = 0.8, marker=dict(
        color=color1,
        line=dict(color='#000000',width=1))
    )

    trace3 =  go.Scatter(
        x=dfx.index,
        y=dfx['Attr%'],
        yaxis = 'y2',
        name=f'% {name}', opacity = 0.6,
        marker=dict(color='black',line=dict(color='#000000',width=0.5))
        )

    layout = dict(
        title=str(cat),
        autosize=False,
        height=500,width=800,
        xaxis=dict(),
        yaxis=dict(title='Count'),
        yaxis2=dict(range=[-0, 75],
                    overlaying='y',
                    anchor='x',
                    side= 'right',
                    zeroline=False,
                    showgrid=False,
                    title=f'% {name}')
        )

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

def plotly_pieplots(
    df:DataFrame,
    cols:ARRN,
    nrows:int,
    ncols:int,
    height:int=800,
    width:int=600,
    title:SN=None,
    colorway:ARRN=None,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False):
    """Multiple pieplots using plotly.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    cols: list
        List of columns.
    nrows: int
        Number of rows in plot.
    ncols: int
        Number of columns in plot.
    height: int
        Figure height.
    width: int
        Figure width.
    title: str
        Main title of the figure.
    colorway: list
        List of colors to use in pieces of pieplots.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.
    """
    # params
    if colorway is None:
        colorway=COLORS30

    if title is None:
        title = 'Pieplots of feature distribution'

    # rows and columns
    nx_ny = [(i+1,j+1) for i in range(nrows) for j in range(ncols)]

    # specs
    specs=[[{'type':'domain'} for i in range(ncols)]
            for _ in range(nrows)]

    # make figure
    fig = make_subplots(rows=nrows, cols=ncols,specs=specs)
    for col,(nx,ny) in zip(cols,nx_ny):
        fig.add_trace(go.Pie(labels=df[col],name=col,title=col),nx, ny)

    fig.update_layout(autosize=False,width=width,height=height)
    fig.update_layout(colorway=colorway)
    fig['layout']['title'] = title
    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

def plotly_scattergl_plot(
    df:DataFrame,
    xcol:str,
    ycol:str,
    color:SN=None,
    colorscale:SN=None,
    logx:bool=False,
    logy:bool=False,
    bestfit:bool=False,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False):
    """Scatterplot for large data using webgl.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    xcol: str
        Name of x column.
    ycol: str
        Name of y column.
    color: str
        Name of color for scatterplot.
    color_scale: str
        Colorscale to choose.
    logx: bool
        Whether or not use log scale in x axis.
    logy: bool
        Whether or not use log scale in y axis.
    best_fit: bool
        Whether or not to show best fit line.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.


    """
    colorcol = df[color] if color in df.columns.values else color
    showscale = True if color in df.columns else False

    xlabel = 'Log({})'.format(xcol.title()) if logx else xcol.title()
    ylabel = 'Log({})'.format(ycol.title()) if logy else ycol.title()

    # x and y
    x = np.log(df[xcol].values) if logx else df[xcol].values
    y = np.log(df[ycol].values) if logy else df[ycol].values


    # get data from scatterplot
    trace = go.Scattergl(
        x=x,
        y=y,
        name = xcol,
        mode='markers',
        marker=dict(opacity=0.5,
                    color=colorcol,
                    showscale=showscale,
                    colorscale=colorscale,
                    colorbar=dict(title='Grade'),),

    )

    data=[trace]

    # data + bestfit when there is bestfit
    if bestfit:
        m,b = np.polyfit(x, y, 1)
        bestfit_y = (m *x + b)

        bestfit=go.Scattergl(
            x=x,
            y=bestfit_y,
            name='Line of best fit',
            line=dict(color='red')
        )

        data= data + [bestfit]

    # layout
    layout = go.Layout(
        title='{} vs. {}'.format(xcol, ycol),
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        hovermode='closest',
        showlegend=False
    )

    fig = go.Figure(data=data, layout=layout)
    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

#===================================================================
def plotly_scattergl_plot_colorcol(
    df:DataFrame,
    xcol:str,
    ycol:str,
    colorcol:str,
    logx:bool=False,
    logy:bool=False,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False):
    """Scatterplot using color column.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    xcol: str
        Name of x column.
    ycol: str
        Name of y column.
    colorcol: str
        Name of column which have color names.
    logx: bool
        Whether or not use log scale in x axis.
    logy: bool
        Whether or not use log scale in y axis.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.


    """
    xlabel = 'Log({})'.format(xcol.title()) if logx else xcol.title()
    ylabel = 'Log({})'.format(ycol.title()) if logy else ycol.title()

    colorcol_lst=sorted(df[colorcol].unique())

    data=[]

    for c in colorcol_lst:
        df_color=df[df[colorcol]==c]

        # x and y
        x = np.log(df_color[xcol].values) if logx else df_color[xcol].values
        y = np.log(df_color[ycol].values) if logy else df_color[ycol].values

        data.append(
            go.Scattergl(
                x=x,
                y=y,
                mode='markers',
                marker=dict(opacity=0.75),
                name= colorcol.title() + ' : '+str(c)))

    layout = go.Layout(
        title='{} vs. {}'.format(xcol.title(), ycol.title()),
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        hovermode='closest',
        showlegend=True)

    fig = go.Figure(data=data, layout=layout)
    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

#============================================================================
def plotly_scattergl_plot_subplots(
    df:DataFrame,
    xcol:str,
    ycol:str,
    subplot_cols:ARRN,
    logx:bool=False,
    logy:bool=False,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False):
    """Plot x vs y scatterplots for all the subplot columns one below another.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    xcol: str
        Name of x column.
    ycol: str
        Name of y column.
    subplot_cols: list
        List of sub plot columns.
    logx: bool
        Whether or not use log scale in x axis.
    logy: bool
        Whether or not use log scale in y axis.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.

    """
    subplots = [i.title() for i in subplot_cols]

    xlabel = 'Log({})'.format(xcol.title()) if logx else xcol.title()
    ylabel = 'Log({})'.format(ycol.title()) if logy else ycol.title()

    fig=make_subplots(rows=5, cols=1, subplot_titles=['Breakup by: '+ col for col in subplots])

    for i in range(len(subplots)):
        col_name=subplots[i]
        col=subplot_cols[i]
        col_values=sorted(df[col].unique())
        for value in col_values:

            df_subset=df[df[col]==value]

            # x and y
            x = np.log(df_subset[xcol].values) if logx else df_subset[xcol].values
            y = np.log(df_subset[ycol].values) if logy else df_subset[ycol].values

            trace=go.Scattergl(
                x=x,
                y=y,
                mode='markers',
                marker=dict(opacity=0.75,),
                name= col_name + ':'+str(value),
                showlegend=False)
            fig.append_trace(trace, i+1, 1)

    fig['layout'].update(
        height=2000,
        title='{} vs. {} - subplots'.format(xcol.title(), ycol.title()),
        hovermode='closest')

    xaxes = [i for i in dir(fig['layout']) if i.startswith('xaxis') ]
    yaxes = [i for i in dir(fig['layout']) if i.startswith('yaxis') ]

    for i in range(len(xaxes)):
        fig['layout'][xaxes[i]]['title'] = xlabel
        fig['layout'][yaxes[i]]['title'] = ylabel

    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

#==============================================================================
def plotly_bubbleplot(
    df1:DataFrame,
    xcol:str,
    ycol1:str,
    ycol2:SN=None,
    size_col:SN=None,
    size_factor:int=5,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False):
    """Bubble plot of two y-axis columns according to size of size column.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    xcol: str
        Name of x column.
    ycol1: str
        Name of y column1.
    ycol2: str
        Name of y column2.
    size_col: str
        Column for size of bubble plot.
    size_factor: int
        Integer to scale size of bubble plot.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.

    Examples
    ---------
    .. code-block:: python

        df1 = df[df.yr_built == 2014]
        plotly_bubbleplot(df1, 'grade', 'bedrooms', 'bathrooms', 'floors', size_factor=5)

    data source: https://www.kaggle.com/harlfoxem/housesalesprediction
    """
    xlabel = xcol.title()
    ylabel =  ycol1.title() + ' & ' +  ycol2.title() if ycol2 else ycol1.title()

    title = "{} vs {} ".format(xcol.title(),ycol1.title())
    if size_col:
        title = "{} vs {} According to {} ".format(xcol.title(),ycol1.title(),size_col.title())

    if ycol2:
        title = "{} vs {} & {} ".format(xcol.title(),ycol1.title(), ycol2.title())

    if (ycol2 and size_col):
        title = "{} vs {} & {} According to {}".format(xcol.title(),ycol1.title(), ycol2.title(),size_col.title())

    marker= dict(size=df1[size_col]*size_factor) if size_col else None

    trace0 = go.Scatter(x=df1[xcol],
                        y=df1[ycol1],
                        name=ycol1.title(),
                        mode="markers",
                        marker = marker
                        )

    data = [trace0]
    if ycol2:
        trace1 = go.Scatter(x=df1[xcol],
                            y=df1[ycol2],
                            name=ycol2.title(),
                            mode="markers",
                            marker=marker
                            )
        data = [trace0,trace1]

    layout = go.Layout(title=title,
                            xaxis=dict(title=xlabel),
                            yaxis=dict(title=ylabel)
                        )

    fig = go.Figure(data=data,layout=layout)
    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)

    if show:
        return iplot(fig)

def get_mapbox_access_token():
    import json
    import os

    with open(os.path.expanduser('~')+ '/.mapbox_tokens.json') as fi:
        json_data = json.load(fi)

    mapbox_access_token = json_data['token1']
    return mapbox_access_token

def plotly_mapbox(
    df1:DataFrame,
    lat_col:str,
    lon_col:str,
    color_col:SN=None,
    text_col:SN=None,
    title:str='My Map',
    marker_size:NUM=4.5,
    zoom:int=9,
    width:int=800,
    height:int=800,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False):
    """Map plot using mapbox.

    Parameters
    -----------
    df1: pandas.DataFrame
        Input data.
    lat_col: str
        Name of latitude column.
    lon_col: str
        Name of longitude column.
    color_col: str
        Name of color column.
    text_col: str
        Name of text column.
    title: str
        Title of the plot.
    marker_size: int
        Size of the marker.
    zoom: int
        Zoom the map.
    width: int
        Width of the map.
    height: int
        Height of the map.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.

    """
    mapbox_access_token = get_mapbox_access_token()

    color = None
    if color_col:
        n = df1[color_col].nunique()
        ncolors = get_distinct_colors(n)
        unq = df1[color_col].unique().tolist()
        mapping = {unq[i]: ncolors[i] for i in range(n)}
        color = df1[color_col].map(mapping)

    if text_col:
        text = df1[text_col]
    elif (color_col and 1):
        text = '(' + df1[lat_col].astype(str) + ',' + df1[lon_col].astype(str) +\
            ') ' + color_col.title() + ': ' + df1[color_col].astype(str)
    else:
        text = df1[lat_col].astype(str) + ',' + df1[lon_col].astype(str)


    lat = df1[lat_col].values.tolist()
    lon = df1[lon_col].values.tolist()

    trace0 = go.Scattermapbox(lat=lat,lon=lon,
                    mode="markers",
                    marker=dict(size=marker_size, color= color) ,
                                hoverinfo="text",
                                text=text)

    layout = dict(title=title,
                width=width,height=height,
                hovermode="closest",
                showlegend=False,
                mapbox=dict(bearing=0,pitch=0,zoom=zoom,
                            center=dict(lat=df1[lat_col].mean(),
                                        lon=df1[lon_col].mean()),
                            accesstoken=mapbox_access_token
                            )
                )

    data = [trace0]
    fig = go.Figure(data=data,layout=layout)

    markdown_html = ' '.join(["""<span style="background:{}"> {} {} </span>""".format(ncolors[i], color_col.title(), unq[i]) for i in range(n)])

    print(markdown_html)
    fig['layout']['title']['x'] = 0.5
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)

#============================ Functions needed ========================
def rgb2hex(color:ARR):
    """Converts a list or tuple of  an RGB values to HEX string.

    Parameters
    -----------
    color: list
        the list or tuple of integers (e.g. (127, 127, 127))

    Returns
    -------
    str:  the rgb string
    """
    return f"#{''.join(f'{hex(int(c))[2:].upper():0>2}' for c in color)}"

def hex_to_rgb(h:str):
    """Convert hexadecimal color codes to rgb

    Parameters
    -----------
    h: str
        Hex string to convert to rgb.

    Examples
    ---------
    .. code-block:: python

        h = '#D8BFD8'
        rgb = 'rgb(216, 191, 216)'
    """
    h = h.lstrip('#')
    rgb = 'rgb' + str(tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))
    return rgb

def get_distinct_colors(key):
    """Get distinct colors.

    Parameters
    -----------
    key: str or int
        String or integer to get the key. Look the example below for valid names.

    Examples
    ---------
    .. code-block:: python

        '''
        key is one of the integers: 10 20 30 40 50 75 100
        key is one of the following strings:

        'reds' 'greens' 'blues' 'grays''yellows'
        'colors10_names' 'colors10_hex'
        'colors20_names' 'colors20_hex'
        'colors30_names' 'colors30_hex'
        'colors40_names' 'colors40_hex'
        'colors50_names' 'colors50_hex',
        'colors75_names' 'colors75_hex'
        'colors100_names' 'colors100_hex'
        '''

    """
    reds = ['#D8BFD8', '#FFC0CB', '#FA8072',
            '#D2691E', '#BC8F8F', '#A52A2A',
            '#FF69B4', '#EE82EE', '#FF1493',
            '#FF00FF', '#DC143C', '#FF0000']

    greens = ['#32CD32', '#006400', '#66CDAA',
            '#20B2AA', '#9ACD32', '#6B8E23',
            '#808000', '#7CFC00', '#90EE90', '#8FBC8F']

    blues=  ['#6495ED', '#B0C4DE', '#0000FF',
            '#00008B', '#6A5ACD', '#800080',
            '#663399', '#8A2BE2', '#DA70D6']

    grays = ['#D3D3D3', '#D8BFD8', '#708090', '#000000']

    yellows = ['#D2B48C', '#FFD700', '#DAA520',
            '#B8860B', '#BDB76B', '#FFFF00']

    colors10_names = ['darkgreen', 'darkblue', 'maroon3',
                    'red', 'yellow', 'lime',
                    'aqua', 'fuchsia', 'cornflower', 'navajowhite']

    colors10_hex = ['#006400', '#00008b', '#b03060',
                    '#ff0000', '#ffff00', '#00ff00',
                    '#00ffff', '#ff00ff', '#6495ed', '#ffdead']

    colors20_names = ['darkslategray', 'maroon', 'darkgreen',
                    'navy', 'yellowgreen', 'red',
                    'darkorange', 'gold', 'lime',
                    'mediumorchid', 'mediumspringgreen', 'aqua',
                    'blue', 'fuchsia', 'dodgerblue',
                    'salmon', 'plum', 'deeppink',
                    'lightskyblue', 'bisque']

    colors20_hex = ['#2f4f4f', '#800000', '#006400',
                    '#000080', '#9acd32', '#ff0000',
                    '#ff8c00', '#ffd700', '#00ff00',
                    '#ba55d3', '#00fa9a', '#00ffff',
                    '#0000ff', '#ff00ff', '#1e90ff',
                    '#fa8072', '#dda0dd', '#ff1493',
                    '#87cefa', '#ffe4c4']

    colors30_names = ['dimgray', 'lightgray', 'saddlebrown',
                    'forestgreen', 'olive', 'darkslateblue',
                    'darkcyan', 'steelblue', 'navy',
                    'darkseagreen', 'darkmagenta', 'maroon3',
                    'orangered', 'darkorange', 'lime',
                    'darkviolet', 'springgreen', 'crimson',
                    'aqua', 'sandybrown', 'blue',
                    'lightcoral', 'greenyellow', 'dodgerblue',
                    'khaki', 'laserlemon', 'plum',
                    'lightgreen', 'deeppink', 'mediumslateblue']

    colors30_hex = ['#696969', '#d3d3d3', '#8b4513',
                    '#228b22', '#808000', '#483d8b',
                    '#008b8b', '#4682b4', '#000080',
                    '#8fbc8f', '#8b008b', '#b03060',
                    '#ff4500', '#ff8c00', '#00ff00',
                    '#9400d3', '#00ff7f', '#dc143c',
                    '#00ffff', '#f4a460', '#0000ff',
                    '#f08080', '#adff2f', '#1e90ff',
                    '#f0e68c', '#ffff54', '#dda0dd',
                    '#90ee90', '#ff1493', '#7b68ee']

    colors40_names = ['darkgray', 'darkslategray', 'darkolivegreen', 'sienna',
                    'forestgreen', 'maroon2', 'midnightblue', 'olive',
                    'darkcyan', 'steelblue', 'yellowgreen', 'darkblue',
                    'limegreen', 'goldenrod', 'darkseagreen', 'purple',
                    'orangered', 'darkorange', 'gold', 'mediumvioletred',
                    'lime', 'crimson', 'aqua', 'deepskyblue',
                    'sandybrown', 'mediumpurple', 'blue', 'purple3',
                    'greenyellow', 'fuchsia', 'dodgerblue', 'palevioletred',
                    'khaki', 'salmon', 'plum', 'paleturquoise',
                    'violet', 'palegreen', 'peachpuff', 'lightpink']

    colors40_hex = ['#a9a9a9', '#2f4f4f', '#556b2f', '#a0522d',
                    '#228b22', '#7f0000', '#191970', '#808000',
                    '#008b8b', '#4682b4', '#9acd32', '#00008b',
                    '#32cd32', '#daa520', '#8fbc8f', '#800080',
                    '#ff4500', '#ff8c00', '#ffd700', '#c71585',
                    '#00ff00', '#dc143c', '#00ffff', '#00bfff',
                    '#f4a460', '#9370db', '#0000ff', '#a020f0',
                    '#adff2f', '#ff00ff', '#1e90ff', '#db7093',
                    '#f0e68c', '#fa8072', '#dda0dd', '#afeeee',
                    '#ee82ee', '#98fb98', '#ffdab9', '#ffb6c1']

    colors50_names = ['darkslategray', 'darkolivegreen', 'olivedrab',
                    'sienna', 'seagreen', 'forestgreen',
                    'maroon2', 'slategray', 'darkslateblue',
                    'rosybrown', 'teal', 'darkgoldenrod',
                    'darkkhaki', 'steelblue', 'navy',
                    'chocolate', 'yellowgreen', 'indianred',
                    'limegreen', 'darkseagreen', 'darkmagenta',
                    'darkorchid', 'orangered', 'orange',
                    'gold', 'mediumblue', 'lime',
                    'mediumspringgreen', 'royalblue', 'crimson',
                    'aqua', 'blue', 'purple3',
                    'greenyellow', 'tomato', 'orchid',
                    'thistle', 'fuchsia', 'palevioletred',
                    'laserlemon', 'cornflower', 'plum',
                    'deeppink', 'lightsalmon', 'wheat',
                    'paleturquoise', 'palegreen', 'lightskyblue',
                    'aquamarine', 'hotpink']

    colors50_hex = ['#2f4f4f', '#556b2f', '#6b8e23', '#a0522d', '#2e8b57',
                    '#228b22', '#7f0000', '#708090', '#483d8b', '#bc8f8f',
                    '#008080', '#b8860b', '#bdb76b', '#4682b4', '#000080',
                    '#d2691e', '#9acd32', '#cd5c5c', '#32cd32', '#8fbc8f',
                    '#8b008b', '#9932cc', '#ff4500', '#ffa500', '#ffd700',
                    '#0000cd', '#00ff00', '#00fa9a', '#4169e1', '#dc143c',
                    '#00ffff', '#0000ff', '#a020f0', '#adff2f', '#ff6347',
                    '#da70d6', '#d8bfd8', '#ff00ff', '#db7093', '#ffff54',
                    '#6495ed', '#dda0dd', '#ff1493', '#ffa07a', '#f5deb3',
                    '#afeeee', '#98fb98', '#87cefa', '#7fffd4', '#ff69b4']

    colors75_names = ['dimgray', 'darkgray', 'gainsboro',
                    'darkslategray', 'darkolivegreen', 'saddlebrown',
                    'olivedrab', 'seagreen', 'forestgreen',
                    'maroon2', 'midnightblue', 'darkgreen',
                    'olive', 'darkslateblue', 'firebrick',
                    'cadetblue', 'lightslategray', 'mediumseagreen',
                    'rosybrown', 'rebeccapurple', 'teal',
                    'darkgoldenrod', 'darkkhaki', 'peru',
                    'steelblue', 'chocolate', 'yellowgreen',
                    'darkblue', 'indigo', 'limegreen',
                    'purple2', 'darkseagreen', 'maroon3',
                    'mediumturquoise', 'mediumaquamarine', 'darkorchid',
                    'orangered', 'orange', 'gold',
                    'yellow', 'mediumvioletred', 'mediumblue',
                    'lawngreen', 'burlywood', 'lime',
                    'mediumorchid', 'mediumspringgreen', 'springgreen',
                    'crimson', 'aqua', 'deepskyblue',
                    'mediumpurple', 'blue', 'purple3',
                    'lightcoral', 'greenyellow', 'lightsteelblue',
                    'coral', 'fuchsia', 'palevioletred',
                    'khaki', 'laserlemon', 'cornflower',
                    'plum', 'powderblue', 'lightgreen',
                    'deeppink', 'mediumslateblue', 'lightsalmon',
                    'violet', 'lightskyblue', 'aquamarine',
                    'hotpink', 'bisque', 'pink']

    colors75_hex = ['#696969', '#a9a9a9', '#dcdcdc', '#2f4f4f', '#556b2f',
                    '#8b4513', '#6b8e23', '#2e8b57', '#228b22', '#7f0000',
                    '#191970', '#006400', '#808000', '#483d8b', '#b22222',
                    '#5f9ea0', '#778899', '#3cb371', '#bc8f8f', '#663399',
                    '#008080', '#b8860b', '#bdb76b', '#cd853f', '#4682b4',
                    '#d2691e', '#9acd32', '#00008b', '#4b0082', '#32cd32',
                    '#7f007f', '#8fbc8f', '#b03060', '#48d1cc', '#66cdaa',
                    '#9932cc', '#ff4500', '#ffa500', '#ffd700', '#ffff00',
                    '#c71585', '#0000cd', '#7cfc00', '#deb887', '#00ff00',
                    '#ba55d3', '#00fa9a', '#00ff7f', '#dc143c', '#00ffff',
                    '#00bfff', '#9370db', '#0000ff', '#a020f0', '#f08080',
                    '#adff2f', '#b0c4de', '#ff7f50', '#ff00ff', '#db7093',
                    '#f0e68c', '#ffff54', '#6495ed', '#dda0dd', '#b0e0e6',
                    '#90ee90', '#ff1493', '#7b68ee', '#ffa07a', '#ee82ee',
                    '#87cefa', '#7fffd4', '#ff69b4', '#ffe4c4', '#ffc0cb']

    colors100_names = ['dimgray', 'darkgray', 'gainsboro',
                    'darkslategray', 'darkolivegreen', 'saddlebrown',
                    'olivedrab', 'seagreen', 'forestgreen',
                    'maroon2', 'midnightblue', 'darkgreen',
                    'olive', 'darkslateblue', 'firebrick',
                    'cadetblue', 'lightslategray', 'mediumseagreen',
                    'rosybrown', 'rebeccapurple', 'teal',
                    'darkgoldenrod', 'darkkhaki', 'peru',
                    'steelblue', 'chocolate', 'yellowgreen',
                    'indianred', 'darkblue', 'indigo',
                    'limegreen', 'purple2', 'darkseagreen',
                    'maroon3', 'tan', 'mediumaquamarine',
                    'darkorchid', 'orangered', 'darkturquoise',
                    'orange', 'gold', 'yellow',
                    'mediumvioletred', 'mediumblue', 'chartreuse',
                    'lime', 'mediumorchid', 'mediumspringgreen',
                    'springgreen', 'royalblue', 'crimson',
                    'aqua', 'deepskyblue', 'mediumpurple',
                    'blue', 'purple3', 'greenyellow',
                    'thistle', 'coral', 'fuchsia',
                    'palevioletred', 'palegoldenrod', 'laserlemon',
                    'cornflower', 'plum', 'lightgreen',
                    'lightblue', 'deeppink', 'lightsalmon',
                    'violet', 'lightskyblue', 'aquamarine',
                    'hotpink', 'bisque', 'pink']

    colors100_hex = ['#696969', '#a9a9a9', '#dcdcdc', '#2f4f4f', '#556b2f',
                    '#8b4513', '#6b8e23', '#2e8b57', '#228b22', '#7f0000',
                    '#191970', '#006400', '#808000', '#483d8b', '#b22222',
                    '#5f9ea0', '#778899', '#3cb371', '#bc8f8f', '#663399',
                    '#008080', '#b8860b', '#bdb76b', '#cd853f', '#4682b4',
                    '#d2691e', '#9acd32', '#cd5c5c', '#00008b', '#4b0082',
                    '#32cd32', '#7f007f', '#8fbc8f', '#b03060', '#d2b48c',
                    '#66cdaa', '#9932cc', '#ff4500', '#00ced1', '#ffa500',
                    '#ffd700', '#ffff00', '#c71585', '#0000cd', '#7fff00',
                    '#00ff00', '#ba55d3', '#00fa9a', '#00ff7f', '#4169e1',
                    '#dc143c', '#00ffff', '#00bfff', '#9370db', '#0000ff',
                    '#a020f0', '#adff2f', '#d8bfd8', '#ff7f50', '#ff00ff',
                    '#db7093', '#eee8aa', '#ffff54', '#6495ed', '#dda0dd',
                    '#90ee90', '#add8e6', '#ff1493', '#ffa07a', '#ee82ee',
                    '#87cefa', '#7fffd4', '#ff69b4', '#ffe4c4', '#ffc0cb']

    colors_dict = {'reds': reds,
                'greens': greens,
                'blues': blues,
                'grays': grays,
                'yellows': yellows,
                'colors10_names': colors10_names,
                'colors10_hex': colors10_hex,
                'colors20_names': colors20_names,
                'colors20_hex': colors20_hex,
                'colors30_names': colors30_names,
                'colors30_hex': colors30_hex,
                'colors40_names': colors40_names,
                'colors40_hex': colors40_hex,
                'colors50_names': colors50_names,
                'colors50_hex' : colors50_hex,
                'colors75_names': colors75_names,
                'colors75_hex': colors75_hex,
                'colors100_names': colors100_names,
                'colors100_hex': colors100_hex,
                10: colors10_hex,
                20: colors20_hex,
                30: colors30_hex,
                40: colors40_hex,
                50: colors50_hex,
                75: colors75_hex,
                100: colors100_hex,
                }
    return colors_dict[key]