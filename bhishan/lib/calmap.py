"""
Calendar heatmaps from Pandas time series data.

Plot Pandas time series data sampled by day in a heatmap per calendar year,
similar to GitHub's contributions calendar.
"""


from __future__ import unicode_literals

import calendar
import datetime

from matplotlib.colors import ColorConverter, ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from distutils.version import StrictVersion

import matplotlib
import matplotlib.cm # we need this, importing only mpl does not work

import plotly.graph_objs as go



__version_info__ = ('0', '0', '7', 'dev')
__date__ = '14 Feb 2016'


__version__ = '.'.join(__version_info__)
__author__ = 'Martijn Vermaat'
__contact__ = 'martijn@vermaat.name'
__homepage__ = 'https://github.com/martijnvermaat/calmap'

_pandas_18 = StrictVersion(pd.__version__) >= StrictVersion('0.18')

# Data used for example:
# all_days = pd.date_range('1/15/2014', periods=700, freq='D')
# days = np.random.choice(all_days, 500)
# events =  pd.Series(np.random.randn(len(days)), index=days)

def yearplot(data, year=None, how='sum', vmin=None, vmax=None, cmap='Reds',
            fillcolor='whitesmoke', linewidth=1, linecolor=None,
            daylabels=calendar.day_abbr[:], dayticks=True,figsize=(12,12),
            monthlabels=calendar.month_abbr[1:], monthticks=True, ax=None,
            **kwargs):
    """
    Plot one year from a timeseries as a calendar heatmap.

    Parameters
    ----------
    data : Series
        Data for the plot. Must be indexed by a DatetimeIndex.
    year : integer
        Only data indexed by this year will be plotted. If `None`, the first
        year for which there is data will be plotted.
    how : string
        Method for resampling data by day. If `None`, assume data is already
        sampled by day and don't resample. Otherwise, this is passed to Pandas
        `Series.resample`.
    vmin, vmax : floats
        Values to anchor the colormap. If `None`, min and max are used after
        resampling data by day.
    cmap : matplotlib colormap name or object
        The mapping from data values to color space.
    fillcolor : matplotlib color
        Color to use for days without data.
    linewidth : float
        Width of the lines that will divide each day.
    linecolor : color
        Color of the lines that will divide each day. If `None`, the axes
        background color is used, or 'white' if it is transparent.
    daylabels : list
        Strings to use as labels for days, must be of length 7.
    dayticks : list or int or bool
        If `True`, label all days. If `False`, don't label days. If a list,
        only label days with these indices. If an integer, label every n day.
    monthlabels : list
        Strings to use as labels for months, must be of length 12.
    monthticks : list or int or bool
        If `True`, label all months. If `False`, don't label months. If a
        list, only label months with these indices. If an integer, label every
        n month.
    ax : matplotlib Axes
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.
    figsize : tuple
        Figure size.
    kwargs : other keyword arguments
        All other keyword arguments are passed to matplotlib `ax.pcolormesh`.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the calendar heatmap.

    Examples
    --------

    By default, `yearplot` plots the first year and sums the values per day:

    .. plot::
        :context: close-figs

        bp.calmap.yearplot(events)

    We can choose which year is plotted with the `year` keyword argment:

    .. plot::
        :context: close-figs

        bp.calmap.yearplot(events, year=2015)

    The appearance can be changed by using another colormap. Here we also use
    a darker fill color for days without data and remove the lines:

    .. plot::
        :context: close-figs

        bp.calmap.yearplot(events, cmap='YlGn', fillcolor='grey',
                        linewidth=0)

    The axis tick labels can look a bit crowded. We can ask to draw only every
    nth label, or explicitely supply the label indices. The labels themselves
    can also be customized:

    .. plot::
        :context: close-figs

        bp.calmap.yearplot(events, monthticks=3, daylabels='MTWTFSS',
                        dayticks=[0, 2, 4, 6])

    """
    if year is None:
        year = data.index.sort_values()[0].year

    if how is None:
        # Assume already sampled by day.
        by_day = data
    else:
        # Sample by day.
        if _pandas_18:
            by_day = data.resample('D').agg(how)
        else:
            by_day = data.resample('D', how=how)

    # Min and max per day.
    if vmin is None:
        vmin = by_day.min()
    if vmax is None:
        vmax = by_day.max()

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    if linecolor is None:
        # Unfortunately, linecolor cannot be transparent, as it is drawn on
        # top of the heatmap cells. Therefore it is only possible to mimic
        # transparent lines by setting them to the axes background color. This
        # of course won't work when the axes itself has a transparent
        # background so in that case we default to white which will usually be
        # the figure or canvas background color.
        linecolor = ax.get_fc()
        if ColorConverter().to_rgba(linecolor)[-1] == 0:
            linecolor = 'white'

    # Filter on year.
    by_day = by_day[str(year)]

    # Add missing days.
    by_day = by_day.reindex(
        pd.date_range(start=str(year), end=str(year + 1), freq='D')[:-1])

    # Create data frame we can pivot later.
    by_day = pd.DataFrame({'data': by_day,
                            'fill': 1,
                            'day': by_day.index.dayofweek,
                            'week': by_day.index.week})

    # There may be some days assigned to previous year's last week or
    # next year's first week. We create new week numbers for them so
    # the ordering stays intact and week/day pairs unique.
    by_day.loc[(by_day.index.month == 1) & (by_day.week > 50), 'week'] = 0
    by_day.loc[(by_day.index.month == 12) & (by_day.week < 10), 'week'] \
        = by_day.week.max() + 1

    # Pivot data on day and week and mask NaN days.
    plot_data = by_day.pivot('day', 'week', 'data').values[::-1]
    plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)

    # Do the same for all days of the year, not just those we have data for.
    fill_data = by_day.pivot('day', 'week', 'fill').values[::-1]
    fill_data = np.ma.masked_where(np.isnan(fill_data), fill_data)

    # Draw heatmap for all days of the year with fill color.
    ax.pcolormesh(fill_data, vmin=0, vmax=1, cmap=ListedColormap([fillcolor]))

    # Draw heatmap.
    kwargs['linewidth'] = linewidth
    kwargs['edgecolors'] = linecolor
    ax.pcolormesh(plot_data, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

    # Limit heatmap to our data.
    ax.set(xlim=(0, plot_data.shape[1]), ylim=(0, plot_data.shape[0]))

    # Square cells.
    ax.set_aspect('equal')

    # Remove spines and ticks.
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.yaxis.set_tick_params(which='both', length=0)

    # Get indices for monthlabels.
    if monthticks is True:
        monthticks = range(len(monthlabels))
    elif monthticks is False:
        monthticks = []
    elif isinstance(monthticks, int):
        monthticks = range(len(monthlabels))[monthticks // 2::monthticks]

    # Get indices for daylabels.
    if dayticks is True:
        dayticks = range(len(daylabels))
    elif dayticks is False:
        dayticks = []
    elif isinstance(dayticks, int):
        dayticks = range(len(daylabels))[dayticks // 2::dayticks]

    ax.set_xlabel('')
    ax.set_xticks([by_day.loc[datetime.date(year, i + 1, 15)].week
                    for i in monthticks])
    ax.set_xticklabels([monthlabels[i] for i in monthticks], ha='center')

    ax.set_ylabel('')
    ax.yaxis.set_ticks_position('right')
    ax.set_yticks([6 - i + 0.5 for i in dayticks])
    ax.set_yticklabels([daylabels[i] for i in dayticks], rotation='horizontal', va='center')

    return ax


def calendarplot(data, how='sum', yearlabels=True,
                yearascending=True, yearlabel_kws=None,
                subplot_kws=None, gridspec_kws=None,
                fig_kws=None, **kwargs):
    """
    Plot a timeseries as a calendar heatmap.

    Parameters
    ----------
    data : Series
        Data for the plot. Must be indexed by a DatetimeIndex.
    how : string
        Method for resampling data by day. If `None`, assume data is already
        sampled by day and don't resample. Otherwise, this is passed to Pandas
        `Series.resample`.
    yearlabels : bool
        Whether or not to draw the year for each subplot.
    yearascending : bool
        Sort the calendar in ascending or descending order.
    yearlabel_kws : dict
        Keyword arguments passed to the matplotlib `set_ylabel` call which is
        used to draw the year for each subplot.
    subplot_kws : dict
        Keyword arguments passed to the matplotlib `add_subplot` call used to
        create each subplot.
    gridspec_kws : dict
        Keyword arguments passed to the matplotlib `GridSpec` constructor used
        to create the grid the subplots are placed on.
    fig_kws : dict
        Keyword arguments passed to the matplotlib `figure` call.
    kwargs : other keyword arguments
        All other keyword arguments are passed to `yearplot`.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
        Tuple where `fig` is the matplotlib Figure object `axes` is an array
        of matplotlib Axes objects with the calendar heatmaps, one per year.

    Examples
    --------

    With `calendarplot` we can plot several years in one figure:

    .. plot::
        :context: close-figs

        calmap.calendarplot(events)

    """
    yearlabel_kws = yearlabel_kws or {}
    subplot_kws = subplot_kws or {}
    gridspec_kws = gridspec_kws or {}
    fig_kws = fig_kws or {}

    years = np.unique(data.index.year)
    if not yearascending:
        years = years[::-1]

    fig, axes = plt.subplots(nrows=len(years), ncols=1, squeeze=False,
                                subplot_kw=subplot_kws,
                                gridspec_kw=gridspec_kws, **fig_kws)
    axes = axes.T[0]

    # We explicitely resample by day only once. This is an optimization.
    if how is None:
        by_day = data
    else:
        if _pandas_18:
            by_day = data.resample('D').agg(how)
        else:
            by_day = data.resample('D', how=how)

    ylabel_kws = dict(
        fontsize=32,
        color=kwargs.get('fillcolor', 'whitesmoke'),
        fontweight='bold',
        fontname='Arial',
        ha='center')
    ylabel_kws.update(yearlabel_kws)

    max_weeks = 0

    for year, ax in zip(years, axes):
        yearplot(by_day, year=year, how=None, ax=ax, **kwargs)
        max_weeks = max(max_weeks, ax.get_xlim()[1])

        if yearlabels:
            ax.set_ylabel(str(year), **ylabel_kws)

    # In a leap year it might happen that we have 54 weeks (e.g., 2012).
    # Here we make sure the width is consistent over all years.
    for ax in axes:
        ax.set_xlim(0, max_weeks)

    # Make the axes look good.
    plt.tight_layout()

    return fig, axes

def plotly_yearplot(df,date,val,woy='woy',
                    day_name='day_name',month_name='month_name',text='text'):
    """
    Plot a timeseries as a yearly heatmap which resembles like github
    contribution plot.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe must have at least two columns `date` and `value`
    date : string
        Name of date column.
    val : string
        Name of value column.
    woy : string, optional
        Column name for week of year.
    day_name : string, optional
        Column name for day name. Eg. Monday or Mon
    month_name : string, optional
        Column name for month name. Eg. January or Jan
    text : string, optional
        Text to display in hoverinfo.

    Returns
    -------
    fig : plotly figure

    Examples
    --------

    .. plot::
        :context: close-figs

        import numpy as np
        import pandas as pd
        from plotly.offline import plot, iplot, init_notebook_mode
        init_notebook_mode(connected=False)

        df = pd.DataFrame({'date': pd.date_range('2020-02-01','2020-12-31',freq='D')})
        df['value'] = np.random.randint(1,20,size=(len(df)))

        from bhishan.calmap import plotly_yearplot
        %load_ext autoreload
        %autoreload 2

        fig = plotly_yearplot(df,'date','value')
        iplot(fig)

    """
    # Required columns
    if not woy: woy = 'woy'
    if not day_name: day_name = 'day_name'
    if not month_name: month_name = 'month_name'
    if not text: text = 'text'

    if woy not in df.columns:
        df = df.copy()
        df[woy] = df[date].dt.weekofyear

    if day_name not in df.columns:
        df[day_name] = df[date].dt.day_name()

    if month_name not in df.columns:
        df[month_name] = df[date].dt.day_name()

    if text not in df.columns:
        df[text] = ('Date: ' + df[date].dt.strftime("%Y %b %d") + '\n' +
            'Value: '+ df[val].astype(str)
            )

    # colorscale
    values = df[val]
    values_dict = values.to_dict()
    norm = matplotlib.colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = matplotlib.cm.get_cmap('Greens')
    greens = ['rgb' + str(cmap(norm(values_dict[i]))[0:3])for i in df[val]]

    data = [
    go.Heatmap(
            x = df[woy],
            y = df[day_name],
            z = df[val],
            text= df[text],
            hoverinfo=text,
            xgap=3, # this
            ygap=3, # and this is used to make the grid-like apperance
            showscale=False,
            colorscale=greens
            )
    ]

    layout = go.Layout(
        title='',
        height=280,
        yaxis=dict(
            showline = False, showgrid = False, zeroline = False,
            tickmode='array',
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            tickvals=[0,1,2,3,4,5,6],
        ),
        xaxis=dict(
            showline=False, showgrid=False, zeroline=False,dtick=4,
        ),
        font={'size':10, 'color':'#9e9e9e'},
        plot_bgcolor=('#fff'),
        margin = dict(t=40),
    )

    fig = go.Figure(data=data, layout=layout)
    return fig
