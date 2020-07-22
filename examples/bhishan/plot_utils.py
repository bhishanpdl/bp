__version__ = 1.0
__author__ = 'Bhishan Poudel'

__doc__ = """
Plot utility functions.

"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import os

home = os.path.expanduser("~")
mpl_path = home + "/Dropbox/a00_Bhishan_Modules/bhishan/mpl_styles/"

def add_text_barplot(ax, decimals=4, rot=30,percent=False,comma=False,
                        fontsize=14):
    """Add the text to bar plot.

    Parameters
    -----------
    ax: matplotlib Axis object
        Matplotlib axis.
    decimals: int
        Number of decimal places to show in top of the bar.
    rot: int
        Rotation to rotate the label on top of the bar.
    percent: bool
        Whether or not show the percentage.
    comma: bool
        Whether or not format the number using comma.

    """
    assert hasattr(ax,'patches')
    for p in ax.patches:
        txt = f"{p.get_height():,.2f}"
        txt = np.round(p.get_height(), decimals=decimals)

        if comma:
            if (int(txt)==txt):
                txt = "{:,}".format(int(txt))
            else:
                txt = "{:,.2f}".format(txt)

        txt = str(txt) + '%' if percent else txt
        x = p.get_x()+p.get_width()/2.
        y = p.get_height()
        ax.annotate(txt, (x,y), ha='center', va='bottom',
                xytext=(0, 10), rotation=rot,
                textcoords='offset points',
                fontsize=fontsize)

def light_axis():
    "Hide the top and right spines"
    ax = plt.gca()
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.xticks(())
    plt.yticks(())
    plt.subplots_adjust(left=.01, bottom=.01, top=.99, right=.99)

def no_axis():
    plt.axis('off')
    plt.subplots_adjust(left=.0, bottom=.0, top=1, right=1)

def _annotate_pearsonr(x, y, **kws):
    from scipy import stats
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("pearsonr = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

def magnify():
    return [dict(selector="th", props=[("font-size", "7pt")]),
            dict(selector="td", props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
            ]

def get_mpl_style(style_name):
    official_styles_lst = plt.style.available
    official_styles_lst = [i for i in official_styles_lst
                            if i!='_classic_test_patch']
    official_styles_dict = dict(enumerate(official_styles_lst))

    custom_styles_dict = {
	"pacoty"       : mpl_path + "pacoty.mplstyle",
    "pitaya_light" : mpl_path + "pitayasmoothie-light.mplstyle",
    "pitaya_dark"  : mpl_path + "pitayasmoothie-dark.mplstyle",
    "qb_common"    : mpl_path + "qb-common.mplstyle",
    "qb_dark"      : mpl_path + "qb-dark.mplstyle",
    "qb_light"     : mpl_path + "qb-light.mplstyle",
    }

    custom_styles_num_dict = {
    # dark custom styles
    -1 : mpl_path + "pitayasmoothie-dark.mplstyle",
    -2 : mpl_path + "qb-common.mplstyle",
    -3 : mpl_path + "qb-dark.mplstyle",
    # light custom styles
	-100 : mpl_path + "pacoty.mplstyle",
    -200 : mpl_path + "pitayasmoothie-light.mplstyle",
    -300 : mpl_path + "qb-light.mplstyle",
    }
    all_styles_names = (official_styles_lst +
                        list(official_styles_dict.keys()) +
                        list(custom_styles_dict.keys()) +
                        list(custom_styles_num_dict.keys())
                        )

    if style_name in official_styles_lst:
        return style_name

    if style_name not in all_styles_names:
        return 'seaborn-darkgrid'

    dics = [official_styles_dict, custom_styles_dict, custom_styles_num_dict ]
    for dic in dics:
        if style_name in dic.keys():
            return dic[style_name]

def get_plotly_colorscale(cmap_name, data):
    import matplotlib as mpl
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    rgb = [sm.to_rgba(i)[:3] for i in data]
    rgb = ["rgb" + str(i)  for i in rgb]

    zero2one = np.linspace(0,1,len(data))

    plotly_colorscale = list(zip(zero2one,rgb))
    return plotly_colorscale

