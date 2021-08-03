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
    ]

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

# local functions
try:
    from .util_colors import rgb2hex, hex_to_rgb,get_distinct_colors
except:
    from util_colors import rgb2hex, hex_to_rgb,get_distinct_colors

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

def plotly_corr(
    df:DataFrame,
    target:SI,
    topN:int=10,
    method:str='pearson',
    colorscale:str='Reds',
    width:int=800,
    height:int=800,
    ytitle:NUM=0.99,
    odir:str='images',
    ofile:str='',
    save:bool=True,
    show:bool=True,
    auto_open:bool=False
    ):
    """Plot correlation heatmap for top N numerical columns.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    target: str
        Target variable name w.r.t which we choose top N other features.
        For example price. Then we get top N features most correlated
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
        Whether or not to automatically open the output html file.

    Examples
    ---------
    .. code-block:: python
        diamonds = sns.load_dataset('diamonds')
        diamonds.bp.plotly_corr('price',topN=4)

    """
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

def plotly_corr_heatmap(
    df:DataFrame,
    target:str,
    topN:int=10,
    method:str='pearson',
    colorscale:str='Reds',
    width:int=800,
    height:int=800,
    ofile:str='',
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
        Whether or not to automatically open the output html file.

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
    col:SI,
    topN:IN=None,
    color:SN=None,
    odir:str='images',
    ofile:str='',
    save:bool=True,
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
        Whether or not to automatically open the output html file.

    Example:
    ----------
    df = sns.load_dataset('tips')
    df.bp.plotly_countplot('day')
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

def plotly_histogram(
    df:DataFrame,
    col:SI,
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

    Examples
    ---------
    .. code-block:: python
        df = sns.load_dataset('diamonds')
        df.bp.plotly_corr('price',topN=4)
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
        Whether or not to automatically open the output html file.

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
    target:SI,
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
        Whether or not to automatically open the output html file.

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
    cols:ARRN=None,
    show_all_points:bool=False,
    width:int=600,
    height:int=800,
    odir:str='images',
    ofile:SN=None,
    show:bool=True,
    save:bool=True,
    auto_open:bool=False
    ):
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
        Whether or not to automatically open the output html file.

    Examples
    ---------
    .. code-block:: python
        titanic = sns.load_dataset('titanic')
        titanic.bp.plotly_boxplot('age')

    Note
    -----
    To display large output in jupyter notebook without scrolling use this:
    .. code-block:: python
        %%javascript
        IPython.OutputArea.auto_scroll_threshold = 9999
    """
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

def plotly_boxplot_allpoints_with_outliers(
    df:DataFrame,
    col:SI,
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
        Whether or not to automatically open the output html file.

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
    xcol:SI,
    ycol:SI,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False
    ):
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
        Whether or not to automatically open the output html file.

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
    cat:SI,
    binn:SI,
    zero:SI,
    one:SI,
    name:str,
    is_one_good:bool=False,
    ofile:str='',
    show:bool=True,
    auto_open:bool=False
    ):
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
        Whether or not to automatically open the output html file.

    Examples
    ---------
    .. code-block:: python

        module_path = "/Users/poudel/Dropbox/a00_Bhishan_Modules/bp/bhishan/"
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
    auto_open:bool=False
    ):
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
        Whether or not to automatically open the output html file.
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

def plotly_yearplot(
    df:DataFrame,
    val:SI,
    date:SI=None,
    index:bool=False,
    cmap:str='Greens',
    text:str='text',
    year:IN=None
    ):
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
    text : string, optional
        Text to display in hoverinfo
    year : int, optional
        Year is required if data is more than one year.

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

def plotly_scattergl_plot(
    df:DataFrame,
    xcol:SI,
    ycol:SI,
    color:SN=None,
    colorscale:SN=None,
    logx:bool=False,
    logy:bool=False,
    bestfit:bool=False,
    ofile:str=None,
    show:bool=True,
    auto_open:bool=False
    ):
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
        Whether or not to automatically open the output html file.
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

def plotly_scattergl_plot_colorcol(
    df:DataFrame,
    xcol:SI,
    ycol:SI,
    colorcol:SI,
    logx:bool=False,
    logy:bool=False,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False
    ):
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
        Whether or not to automatically open the output html file.


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

def plotly_scattergl_plot_subplots(
    df:DataFrame,
    xcol:SI,
    ycol:SI,
    subplot_cols:ARRN,
    logx:bool=False,
    logy:bool=False,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False
    ):
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
        Whether or not to automatically open the output html file.

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

def plotly_bubbleplot(
    df1:DataFrame,
    xcol:SI,
    ycol1:SI,
    ycol2:SN=None,
    size_col:SN=None,
    size_factor:int=5,
    ofile:SN=None,
    show:bool=True,
    auto_open:bool=False
    ):
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
        Whether or not to automatically open the output html file.

    Examples
    ---------
    .. code-block:: python

        df = sns.load_dataset('titanic')
        bp.plotly_bubbleplot(df, 'age', 'fare','pclass', size_factor=5)

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
