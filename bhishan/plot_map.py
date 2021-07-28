__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various functions for map plotting.

- plotly_usa_map(df,col_state,col_value)
- plotly_usa_map2(df,col_state,col_value)
- plotly_agg_usa_plot(df,col_state,col_value)
- plotly_usa_bubble_map(df,col_value,col_lat,col_lon)
- plotly_country_plot(df1,col_country,col_value)
- plotly_agg_country_plot(df,col_country,col_value)
- plotly_mapbox(df1, lat_col, lon_col)

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.plot_map?

"""
__all__ = ["plotly_usa_map",
        "plotly_usa_map2",
        "plotly_agg_usa_plot",
        "plotly_usa_bubble_map",
        "plotly_country_plot",
        "plotly_agg_country_plot",
        "plotly_mapbox"
        ]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.tools import make_subplots
import plotly.tools as tls
from plotly.offline import plot, iplot


import plotly.express as px
# Plotly Notes
"""
Colorscales: https://plot.ly/python/builtin-colorscales/
The default colorscale is ‘RdBu’.

['Reds','Greens','Blues','Greys',
'Viridis','Earth','Rainbow',
'Blackbody','Bluered','Electric',
'Hot','Jet',
'Picnic','Portland',
'RdBu','YlGnBu','YlOrRd']

Custom colorscales:
colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(255, 0, 0)']]


New line in plotly:
df['text'] = df['state'] + '<br>' + df['city']
""";

def plotly_usa_map2(df,col_state,col_value,title=None,
                ofile=None,show=True,auto_open=False):

    """Plotly map plot for different states of USA.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col_state: str
        Column name of states.
    col_value: str
        Column name of values.
    title: str
        title of the plot.
    ofile: str
        Name of the output file.
    show: bool
        Whether or not to show the rendered html in notebook.
    auto_open: bool
        Whether or not to automatically open the ouput html file.


    Example 1
    -----------
    df = pd.DataFrame({'state': ['NY', 'OH','MI','CA','TX'],
                    'value': [100,200,300,400,500]})
    plotly_usa_map2(df,'state','value')

    """

    # make sure df <= 50
    assert len(df) <= 50

    fig = px.choropleth(df,
                        locations=col_state,
                        color=col_value,
                        hover_name=col_state,
                        locationmode = 'USA-states')

    fig.update_layout(
        title_text = title,
        geo_scope='usa',
    )
    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)

    if show:
        iplot(fig, validate=False)

def plotly_usa_map(df,col_state,col_value,col_text=None,
        colorscale='Viridis',reversescale=False,
        title=None,
        width=800,height=800,
        ofile=None,show=True,
        auto_open=False):
    """Plotly map plot for different states of USA.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col_state: str
        Name of State column. eg. OH
    col_value: str
        Name of Value column. e.g. mean_salary
    col_text: str
        Name of Text column.
    colorscale: str
        Valid plotly colorscale name.
        The default colorscale is ‘RdBu’.
        https://plot.ly/python/builtin-colorscales/
        One of the following colorscales:
        ['Reds','Greens','Blues','Greys',
        'Viridis','Earth','Rainbow',
        'Blackbody','Bluered','Electric',
        'Hot','Jet',
        'Picnic','Portland',
        'RdBu','YlGnBu','YlOrRd']
    reversescale: bool
        Whether or not to reverse the colorscale.
    title: str
        Title of the plot.
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

    Example 1
    -----------
    df = pd.DataFrame({'state': ['NY', 'OH','MI','CA','TX'],
                    'value': [100,200,300,400,500]})
    plotly_usa_map(df,'state','value')
    """
    # make sure df <= 50
    assert len(df) <= 50

    # title
    if not title:
        title = col_value + ' per state of USA'
    # text
    if not col_text:
        text = df[col_state]

    data = [dict(
                type='choropleth',
                colorscale = colorscale,
                reversescale=reversescale,
                locations=df[col_state],
                z=df[col_value],
                text=text,
                locationmode='USA-states'
                )
            ]

    layout = dict(title = title,
                height = height,
                width  = width,
                geo = dict(scope='usa',
                            projection=dict(type ='albers usa'),
                            showlakes = True,
                            lakecolor = 'rgb(255,255,255)')
                )

    # figure
    fig = dict(data=data, layout=layout )

    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)

    if show:
        iplot(fig, validate=False)

# USA States Plot Aggregations
def plotly_agg_usa_plot(df,col_state,col_value,
                            xlabel='x',ylabel='y',ylim=None,
                            colorscale = 'Portland',
                            reversescale=False,
                            title=None,
                            width=800,height=800,
                            ofile=None,show=True,
                            auto_open=False):
    """Plotly map plot for different states in USA.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col_state: str
        Name of State column.
    col_value: str
        Name of Value column.eg. HappinessScore
    xlabel: str
        X axis label name.
    ylabel: str
        Y axis label name.
    ylim: list
        Y-axis limit. eg. [0,22]
    colorscale: str
        Valid plotly colorscale name.
        The default colorscale is ‘RdBu’.
        https://plot.ly/python/builtin-colorscales/
        One of the following colorscales:
        ['Reds','Greens','Blues','Greys',
        'Viridis','Earth','Rainbow',
        'Blackbody','Bluered','Electric',
        'Hot','Jet',
        'Picnic','Portland',
        'RdBu','YlGnBu','YlOrRd']
    reversescale: bool
        Whether or not to reverse the colorscale.
    title: str
        Title of the plot.
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


    Example
    --------
    df = pd.DataFrame({'state': ['NY', 'OH','MI','CA','TX'],
                        'value': [100,200,300,400,500]})

    plotly_agg_usa_plot(df,'state','value',width=600)

    """
    # title
    if not title:
        title = col_value + ' per State'


    aggs = ["count","sum","avg","median","mode",
            "rms","stddev","min","max","first","last"]

    agg = []
    agg_func = []
    for i in range(0, len(aggs)):
        agg = dict(
            args=['transforms[0].aggregations[0].func', aggs[i]],
            label=aggs[i],
            method='restyle'
        )
        agg_func.append(agg)

    data = [dict(
            type = 'choropleth',
            locationmode = 'USA-states',
            locations = df[col_state],
            z = df[col_value],
            autocolorscale = False,
            colorscale = colorscale,
            reversescale = reversescale,
            showcolorbar=True,
            transforms = [dict(
                type = 'aggregate',
                groups = df[col_state],
                aggregations = [dict(
                    target = 'z',
                    func = 'sum',
                    enabled = True)
                ]
            )]
    )]

    layout = dict(
        title = title,
        xaxis = dict(title = xlabel),
        yaxis = dict(title = ylabel, range = ylim),
        height = height,
        width = width,
        updatemenus = [dict(x = 1.15,y = 1.15,
            xref = 'paper',yref = 'paper',
            yanchor = 'top',active = 1,
            showactive = False,
            buttons = agg_func
        )],
        geo = dict(
            showframe = True,
            scope = 'usa',
            showcoastlines = True,
            showlakes = True,
            lakecolor = 'rgb(255,255,255)',
            projection = dict(type = 'albers usa')
        )
    )


    # figure
    fig = {'data': data,'layout': layout}

    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)

    if show:
        iplot(fig, validate=False)

def plotly_usa_bubble_map(df,col_value,col_lat,col_lon,
        limits,colors,
        scale=5000,
        col_text=None,
        title=None,
        width=800,height=800,
        ofile=None,show=True,
        auto_open=False):
    """Plotly map plot for different states of USA.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col_value: str
        Name of Value column. e.g. mean_salary
    col_lat: str
        Name of Latitude column. eg. lat
    col_lon: str
        Name of Longitude column. e.g. long
    limits: list
        List of limits.
    scale: int
        Scale of limit.
    col_text: str
        Name of Text column.
    colors: list
        List of colors for limits.
    title: str
        Title of the plot.
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

    Example 1
    -----------
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv')


    df['text'] = df['name'] + '<br>Population ' +\
                    (df['pop']/1e6).astype(str)+' million'

    limits = [(0,2),(3,10),(11,20),(21,50),(50,3000)]
    colors = ["royalblue","crimson","lightseagreen",
                "orange","lightgrey"]


    plotly_usa_bubble_map(df,'pop','lat','lon',
            limits,colors,
            scale=5000,
            col_text='text')
    """

    # title
    if not title:
        title = col_value
    # text
    if not col_text:
        text = df[col_state]

    cities = []
    scale = scale

    fig = go.Figure()

    for i in range(len(limits)):
        lim = limits[i]
        df_sub = df[lim[0]:lim[1]]
        fig.add_trace(go.Scattergeo(
            locationmode = 'USA-states',
            lon = df_sub[col_lon],
            lat = df_sub[col_lat],
            text = df_sub[col_text],
            marker = dict(
                size = df_sub[col_value]/scale,
                color = colors[i],
                line_color='rgb(40,40,40)',
                line_width=0.5,
                sizemode = 'area'
            ),
            name = '{0} - {1}'.format(lim[0],lim[1])))

    fig.update_layout(
            height = height,
            width = width,
            title_text = title,
            showlegend = True,
            geo = dict(
                scope = 'usa',
                landcolor = 'rgb(217, 217, 217)',
            )
        )

    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)

    if show:
        iplot(fig, validate=False)

#================================= Countries Plot ===================
# Country Plot
def plotly_country_plot(df1,col_country,col_value,
                            xlabel='x',ylabel='y',ylim=None,
                            colorscale='Portland',
                            reversescale=False,
                            title=None,
                            width=800,height=800,
                            ofile=None,show=True,
                            auto_open=False):
    """Plotly map plot for different countries.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col_country: str
        Name of Country column.
    col_value: str
        Name of Value column.eg. HappinessScore
    xlabel: str
        X axis label name.
    ylabel: str
        Y axis label name.
    ylim: list
        Y-axis limit. eg. [0,22]
    colorscale: str
        Valid plotly colorscale name.
        The default colorscale is ‘RdBu’.
        https://plot.ly/python/builtin-colorscales/
        One of the following colorscales:
        ['Reds','Greens','Blues','Greys',
        'Viridis','Earth','Rainbow',
        'Blackbody','Bluered','Electric',
        'Hot','Jet',
        'Picnic','Portland',
        'RdBu','YlGnBu','YlOrRd']
    reversescale: bool
        Whether or not to reverse the colorscale.
    title: str
        Title of the plot.
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


    Example 1
    -----------
    df1 = pd.DataFrame({'country': ['USA', 'Canada','China','Russia','Brazil'],
                        'count': [100,200,300,400,500]})

    Example 2
    -----------
    df1 = df['Year'].groupby(df['Birth Country']).count().reset_index()
                    .sort_values('Year', ascending = False)

    Example 3
    -------------
    note: production_countries = ['USA','Japan'], [], ['Brazil'] etc
        i.e. the column has list instead of direct country names.

    import itertools
    df1 = (pd.Series(list(itertools.chain.from_iterable(movie['production_countries'])))
            .value_counts()
            .reset_index()
            .rename(columns={'index':'country', 0:'num_movies'})
            )
    """
    # assertions
    assert len(df) <= 200

    # title
    if not title:
        title = col_value + ' per Country'

    data = [ dict(
            type           = 'choropleth',
            locations      = df1[col_country],
            locationmode   = 'country names',
            z              = df1[col_value],
            text           = df1[col_country],
            colorscale     = colorscale,
            autocolorscale = False,
            reversescale   = reversescale,
        ) ]

    layout = dict(
        title  = title,
        xaxis  = dict(title = xlabel),
        yaxis  = dict(title = ylabel, range = ylim),
        height = height,
        width  = width,
        geo    = dict(showframe = True,showcoastlines = True,
                        projection = dict(type = 'mercator')
        )
    )

    # figure
    fig = dict(data=data, layout=layout )

    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)

    if show:
        iplot(fig, validate=False)
#================================= map plot with agg ===========
# Country Plot Aggregations
def plotly_agg_country_plot(df,col_country,col_value,
                            xlabel='x',ylabel='y',ylim=None,
                            colorscale='Portland',
                            reversescale=False,
                            title=None,
                            width=800,height=800,
                            ofile=None,show=True,
                            auto_open=False):
    """Plotly map plot for different countries.

    Parameters
    -----------
    df: pandas.DataFrame
        Input data.
    col_country: str
        Name of Country column.
    col_value: str
        Name of Value column.eg. HappinessScore
    xlabel: str
        X axis label name.
    ylabel: str
        Y axis label name.
    ylim: list
        Y-axis limit. eg. [0,22]
    colorscale: str
        Valid plotly colorscale name.
        The default colorscale is ‘RdBu’.
        https://plot.ly/python/builtin-colorscales/
        One of the following colorscales:
        ['Reds','Greens','Blues','Greys',
        'Viridis','Earth','Rainbow',
        'Blackbody','Bluered','Electric',
        'Hot','Jet',
        'Picnic','Portland',
        'RdBu','YlGnBu','YlOrRd']
    reversescale: bool
        Whether or not to reverse the colorscale.
    title: str
        Title of the plot.
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


    Example
    --------
    df = pd.read_csv("https://raw.githubusercontent.com/bcdunbar/datasets/master/worldhappiness.csv")
    plotly_agg_country_plot(df,'Country','HappinessScore',width=600)

    """
    # title
    if not title:
        title = col_value + ' per Country'

    aggs = ["count","sum","avg","median","mode",
            "rms","stddev","min","max","first","last"]

    agg = []
    agg_func = []
    for i in range(0, len(aggs)):
        agg = dict(
            args=['transforms[0].aggregations[0].func', aggs[i]],
            label=aggs[i],
            method='restyle'
        )
        agg_func.append(agg)

    data = [dict(
        type = 'choropleth',
        locationmode = 'country names',
        locations = df[col_country],
        z = df[col_value],
        autocolorscale = False,
        colorscale = colorscale,
        reversescale = reversescale,
        transforms = [dict(
        type = 'aggregate',
        groups = df[col_country],
        aggregations = [dict(
            target = 'z', func = 'sum', enabled = True)
        ]
        )]
    )]

    layout = dict(
        title = title,
        xaxis = dict(title = xlabel),
        yaxis = dict(title = ylabel, range = ylim),
        height = height,
        width = width,
        updatemenus = [dict(x = 1.15,y = 1.15,
            xref = 'paper',yref = 'paper',
            yanchor = 'top',active = 1,
            showactive = False,
            buttons = agg_func
        )]
    )

    # figure
    fig = {'data': data,'layout': layout}

    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)

    if show:
        iplot(fig, validate=False)

#========================= mapbox
def get_mapbox_access_token():
    import json
    import os

    with open(os.path.expanduser('~')+ '/.mapbox_tokens.json') as fi:
        json_data = json.load(fi)

    mapbox_access_token = json_data['token1']
    return mapbox_access_token

def plotly_mapbox(df1, lat_col, lon_col, color_col=None, text_col=None,
                title='My Map',
                marker_size=4.5,zoom=9,
                width=800,height=800,
                ofile=None,show=True,
                auto_open=False):
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

    trace0 = go.Scattermapbox(lat=lat,
                            lon=lon,
                            mode="markers",
                            marker=dict(size=marker_size, color= color) ,
                            hoverinfo="text",
                            text=text)

    layout = dict(title=title,
        width=width,
        height=height,
        hovermode="closest",
        showlegend=False,
        mapbox=dict(bearing=0,
                    pitch=0,
                    zoom=zoom,
                    center=dict(lat=df1[lat_col].mean(),
                                lon=df1[lon_col].mean()),
                                accesstoken=mapbox_access_token))

    data = [trace0]
    fig = go.Figure(data=data,layout=layout)

    markdown_html = ' '.join(["""<span style="background:{}"> {} {} </span>""".format(ncolors[i], color_col.title(), unq[i]) for i in range(n)])

    print(markdown_html)

    if ofile:
        plot(fig, filename=ofile,auto_open=auto_open)
    if show:
        return iplot(fig)