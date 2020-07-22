__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various bokeh plotting utilities.

- bokeh_scatterplot(df,xcol,ycol,ofile=None
- bokeh_countplot(df, cat,ofile=None,height=300,width=400)
- bokeh_stacked_barplot(df, xcol,ycols,ofile=None)
- bokeh_stacked_countplot(df, xcol,ycol,target
- bokeh_histogram(df,col,n_bins,bin_range=None,ofile=None)
- bokeh_merc(df, col_lat, col_lon)
- bokeh_map_plot(df, col_lat, col_lon,ofile=None)

To stop auto opening of new tab with html in jupyte notebook, we can use
following two lines in each cell of jupyter notebook:

.. code-block:: python

    import bokeh
    bokeh.io.reset_output()
    bokeh.io.output_notebook()

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.plot_bokeh?

"""
__all__ = ['bokeh_scatterplot',
        'bokeh_countplot',
        'bokeh_stacked_barplot',
        'bokeh_stacked_countplot',
        'bokeh_histogram',
        'bokeh_merc',
        'bokeh_map_plot',
        ]


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh
from bokeh.io import output_file, output_notebook, save
from bokeh.plotting import figure, show, reset_output
from bokeh.models import ColumnDataSource
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel

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

colors = reds + greens + blues + grays + yellows
colors = colors * 100


#===============================================================================
# Scatterplot
#===============================================================================
def bokeh_scatterplot(df,xcol,ycol,ofile=None,
        xaxis_type='linear',yaxis_type='linear',
        xrange=None,yrange=None,
        height=300,width=400,
        color='#000000',size=10):
    """Scatterplot using bokeh.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe.
    xcol: str
        x column name.
    ycol: str
        y column name.
    ofile: str
       Name of output html.
    xaxis_type: str
        Type of xaxis. e.g. linear, log.
    yaxis_type: str
        Type of yaxis. Eg. linear, log.
    xrange: list
       X axis limit.
    yrange: list
        Y axis limit.
    height: int
        Height of the image.
    width: int
        Width of the image.
    color: str
        Name of color. e.g hexstrings.
    size: int
       Size of the scatter dots.


    """
    # if ofile write html and open new tab.
    if ofile:
        output_file(ofile)
        height = 800
        width = 800

    # when ofile is None, do not open new tab.
    if not ofile:
            bokeh.io.reset_output()
            bokeh.io.output_notebook()

    source = ColumnDataSource(df)
    p = figure(title="{} vs. {} Scatterplot".format(xcol.title(), ycol.title()),
              x_axis_label=xcol.title(),
              y_axis_label=ycol.title(),
              plot_height=height,
              plot_width=width,
              x_axis_location='below',
              x_axis_type=xaxis_type,
              y_axis_type=yaxis_type,
              x_range=xrange,
              y_range=yrange)
    p.scatter(x=xcol, y=ycol, line_color=color, source=source, size=size)

    show(p)

#===============================================================================
# Count plot
#===============================================================================
def bokeh_countplot(df, cat,ofile=None,height=300,width=400):
    """Count plot of categorical feature using bokeh.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.

    cat: str
        categorical column name.

    Examples
    ---------
    If the figure is not displayed use following:
    .. code-block:: python

        import bokeh
        bokeh.io.reset_output()
        bokeh.io.output_notebook()

    """
    from bokeh.io import show, output_file
    from bokeh.palettes import Category20_20
    from bokeh.models import ColumnDataSource
    from bokeh.plotting import figure

    # if ofile write html and open new tab.
    if ofile:
        output_file(ofile)
        height = 800
        width = 800

    # when ofile is None, do not open new tab.
    if not ofile:
            bokeh.io.reset_output()
            bokeh.io.output_notebook()

    vals = df[cat].value_counts().index.values.astype(str).tolist()
    counts = df[cat].value_counts().values.tolist()
    colors = (Category20_20*10)[:len(vals)]

    source = ColumnDataSource(data=dict(vals=vals,
                                        counts=counts,
                                        color=colors))

    p = figure(x_range=vals,y_range=None,
            title= cat.title() + " Counts",
            x_axis_label=cat.title(),
            y_axis_label='Count',
            plot_height=height,
            plot_width=width,
            toolbar_location=None,
            tools="hover",
            tooltips="Count: @counts "
            )

    p.vbar(x='vals', top='counts', width=0.9,
        color='color', legend="vals", source=source)

    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"

    show(p)

#===============================================================================
# Stacked Bar plot
#===============================================================================
def bokeh_stacked_barplot(df, xcol,ycols,ofile=None):
    """Stacked barplot using bokeh.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    xcol: str
        Name of x column.
    ycols: list
        List with y column names.
    ofile: str
        Name of output file.

    Examples
    ---------
    .. code-block:: python

        df1 = pd.DataFrame({'fruits' : ['Apples', 'Pears', 'Nectarines',
                                    'Plums', 'Grapes', 'Strawberries'],
                '2015'   : [2, 1, 4, 3, 2, 4],
                '2016'   : [5, 3, 4, 2, 4, 6],
                '2017'   : [3, 2, 4, 4, 5, 3]})

        ycols = ['2015','2016']
        xcol = 'fruits'
        stacked_barplot_bokeh(df1, 'fruits',['2015','2016'])


        df1 = (df.groupby(['yr_sales','view'])
            .agg({'price':'count'})
            .unstack(level=0)
            .droplevel(0,axis=1)
            .reset_index()
            )

        stacked_barplot_bokeh(df1, 'view',df1.columns[1:])

    """
    import bokeh
    from bokeh.palettes import Category20
    from bokeh.plotting import figure
    from bokeh.core.properties import value
    from bokeh.io import show, output_file, reset_output, output_notebook

    # we need to make column names string.
    df.columns = df.columns.astype(str)
    ycols = np.array(ycols).astype(str)

    colors = (Category20[20]*10)[:len(ycols)]
    xvals = df[xcol].values.astype(str).tolist()

    # if ofile write html and open new tab.
    if ofile:
        output_file(ofile)
        height = 800
        width = 800

    # when ofile is None, do not open new tab.
    if not ofile:
            bokeh.io.reset_output()
            bokeh.io.output_notebook()

    # plot figure
    p = figure(x_range=xvals,
            plot_height=250, title="",
            toolbar_location=None,
            tools="hover",
            tooltips="$name @{}: @$name".format(xcol))

    # vertical bar stacks
    p.vbar_stack(ycols, x=xcol, width=0.9, color=colors, source=df,
                legend=[value(x) for x in ycols])

    # settings
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"

    show(p)
#===============================================================================
# Stacked count plot
#===============================================================================
def bokeh_stacked_countplot(df, xcol,ycol,target,
    ofile=None,agg_type='count',height=300,width=400):
    """Stacked countplot using bokeh.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    xcol: str
        Name of x column.
    ycol: str
        Name of y column.
    target: str
        Name of the target column.
    ofile: str
        Name of the output file.
    agg_type: str
        Type of aggregation. E.g. 'count', 'sum'
    height: int
        Height of the plot.
    width: int
        Width of the plot.

    """
    import bokeh
    from bokeh.palettes import Category20
    from bokeh.plotting import figure
    from bokeh.core.properties import value
    from bokeh.io import show, output_file, reset_output, output_notebook

    # get the tidy format dataframe
    df1 = (df.groupby([ycol,xcol])
        .agg({target:agg_type})
        .unstack(level=0)
        .droplevel(0,axis=1)
        .reset_index()
        )
    # if ofile write html and open new tab.
    if ofile:
        output_file(ofile)
        height = 800
        width = 800

    # when ofile is None, do not open new tab.
    if not ofile:
            bokeh.io.reset_output()
            bokeh.io.output_notebook()

    # we need to make column names string.
    df1.columns = df1.columns.astype(str)
    ycols = df1.columns[1:]

    colors = (Category20[20]*10)[:len(ycols)]
    xvals = df1[xcol].values.astype(str).tolist()

    # plot figure
    p = figure(x_range=xvals,
            title= "Stacked count plot of {} vs {}".format(
                xcol.title(), ycol.title()
            ),
            x_axis_label=xcol.title(),
            y_axis_label=agg_type.title(),
            plot_height=height,
            plot_width=width,
            toolbar_location=None,
            tools="hover",
            tooltips="$name @{}: @$name".format(xcol))

    # vertical bar stacks
    p.vbar_stack(ycols, x=xcol, width=0.9, color=colors, source=df1,
                legend=[value(x) for x in ycols])

    # settings
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"

    show(p)

#===============================================================================
# Histogram
#===============================================================================
def bokeh_histogram(df,col,n_bins,bin_range=None,ofile=None,
                        height=300,width=400,
                        color='violet'):
    """Plot interactive histogram using bokeh.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col: str
        Name of the column.
    n_bins: int
        Number of bins in the histogram.
    bin_range: (float, float)
        Lower and uppr range of bins in the histogram.
    ofile: str
        Name of the output file.
    height: int
        Height of the plot.
    width: int
        Width of the plot.
    color: str
        Color of the histogram.

    """
    import pandas as pd
    import numpy as np
    from bokeh.plotting import figure

    from bokeh.io import show, output_notebook
    from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
    from bokeh.palettes import Category10_5, Category20_16

    # some labels
    title = "Histogram of {}".format(col.title())
    x_axis_label = col.title()
    x_tooltip = col.title()

    # if ofile write html and open new tab.
    if ofile:
        output_file(ofile)
        height = 800
        width = 800

    # when ofile if None, do not open new tab.
    if not ofile:
            bokeh.io.reset_output()
            bokeh.io.output_notebook()

    # histogram from numpy.histogram
    arr_hist, edges = np.histogram(df[col].values,bins=n_bins,range=bin_range)

    # pandas dataframe from numpy histogram values
    arr_df = pd.DataFrame({'count': arr_hist,
                           'left': edges[:-1],
                           'right': edges[1:]})

    arr_df['f_count'] = ['%d' % count for count in arr_df['count']]
    arr_df['f_interval'] = ['%d to %d ' % (left, right)
                            for left, right in zip(arr_df['left'], arr_df['right'])]

    # column data source from pandas dataframe
    arr_src = ColumnDataSource(arr_df)

    # Set up the figure same as before
    p = figure(plot_width = width,
               plot_height = height,
               title = title,
               x_axis_label = x_axis_label,
               y_axis_label = 'Count')

    # Add a quad glyph with source this time
    p.quad(bottom=0,
           top='count',
           left='left',
           right='right',
           source=arr_src,
           fill_color=color,
           hover_fill_alpha=0.7,
           hover_fill_color='blue',
           line_color='black')

    # Add style to the plot
    p.title.align = 'center'
    p.title.text_font_size = '18pt'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'

    # Add a hover tool referring to the formatted columns
    hover = HoverTool(tooltips = [(x_tooltip, '@f_interval'),
                                  ('Count', '@f_count')])

    # Add the hover tool to the graph
    p.add_tools(hover)

    # show the plot
    show(p)

#===============================================================================
# Map visualization
#===============================================================================
def bokeh_merc(df, col_lat, col_lon):
    """Mercaptor plot.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col_lat: str
        Name of latitude column.
    col_lon: str
        Name of longitude column.

    """
    lat = df[col_lat]
    lon = df[col_lon]

    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x/lon
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lat * (np.pi/180.0)/2.0)) * scale
    return (x, y)

def bokeh_map_plot(df, col_lat, col_lon,ofile=None):
    """Mercaptor plot from latitude and longitude using bokeh.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col_lat: str
        Name of latitude column.
    col_lon: str
        Name of longitude column.
    ofile: str
        Name of output file.

    """
    from bokeh.io import output_file, output_notebook
    from bokeh.plotting import figure, show
    from bokeh.tile_providers import get_provider, Vendors

    tile_provider = get_provider(Vendors.CARTODBPOSITRON)

    # if ofile write html and open new tab.
    if ofile:
        output_file(ofile)
        height = 800
        width = 800

    # when ofile if None, do not open new tab.
    if not ofile:
            bokeh.io.reset_output()
            bokeh.io.output_notebook()

    p = figure(x_axis_type="mercator", y_axis_type="mercator")
    p.add_tile(tile_provider)

    p.circle(x = merc(df, col_lat, col_lon)[0],
             y = merc(df, col_lat, col_lon)[1]
            )

    show(p)
