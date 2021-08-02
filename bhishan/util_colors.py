__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various utilities for plotting.

- rgb2hex(color)
- hex_to_rgb(h)
- get_distinct_colors(key) key is 10 20 30 40 50 75 100
    or, key is reds greens colors10_hex colors10_names etc.
- discrete_cmap(N, base_cmap=None)

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.plot_colors?

"""
__all__ = [
    "rgb2hex",
    "hex_to_rgb",
    "get_distinct_colors",
    "discrete_cmap",
    "get_colornames_from_cmap"
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

def rgb2hex(color:TL)->str:
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

def hex_to_rgb(h:str)->str:
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

def get_distinct_colors(key:SI)-> LD:
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

def discrete_cmap(N:int,
    base_cmap:Any=None)-> Any:
    """Create an N-bin discrete colormap from the specified input map
    Reference: https://gist.github.com/jakevdp/91077b0cae40f8f8244a

    Example:
    N = 5
    x = np.random.randn(40)
    y = np.random.randn(40)
    c = np.random.randint(N, size=40)

    # plot
    plt.scatter(x, y, c=c, s=50, cmap=discrete_cmap(N, 'cubehelix'))
    plt.colorbar(ticks=range(N))
    plt.clim(-0.5, N - 0.5)
    plt.show()

    """
    from matplotlib.colors import LinearSegmentedColormap
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return LinearSegmentedColormap.from_list(cmap_name, color_list, N)

def get_colornames_from_cmap(cmap_name:str,n:int=10)-> List:
    """Get string of color names from given matplotlib cmap name.

    Example:
    cmap_name = 'cubehelix'
    n = 10
    colors = get_colornames_from_cmap(cmap_name, n)

    """
    import matplotlib

    cmap = matplotlib.cm.get_cmap(cmap_name, n)
    color_names = [matplotlib.colors.rgb2hex(cmap(i))
                    for i in range(cmap.N)]
    return color_names