__author__ = 'Bhishan Poudel'

__doc__ = """
This module prints the help htmls or markdowns I have stored in Dropbox.

- hlp(q)

Usage
-------
.. code-block:: python

    %ext_load autoreload
    %autoreload 2

    from bhishan import bp
    bp.hlp('cohens_q')

"""

__all__ = ["hlp"]
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
import os
import sys
import time
import glob
from IPython.display import display, HTML,Markdown

home = os.path.expanduser('~') + '/'
help_dir = home + 'Dropbox/a00_Bhishan_Modules/bhishan/help/'

help_htmls = glob.glob(help_dir + '*.html')
help_html_bases = [ i.rstrip('.html').split('/')[-1] for i in help_htmls]

help_mds = glob.glob(help_dir + '*.md')
help_md_bases = [ i.rstrip('.md').split('/')[-1] for i in help_mds]


def hlp(q:str):
    """Display help htmls in Jupyter notebook from Dropbox help folder.

    Parameters
    -----------
    q: str
        Base name of html or markdown stored in my help dir.
        e.g. 'cohens_d', 'cohens_q', 'conda', 'eta_squared'

    Usage
    -------
    .. code-block:: python

        %ext_load autoreload
        %autoreload 2

        from bhishan import bp
        bp.hlp('cohens_q')

    """
	# html
    path_html = 'Dropbox/a00_Bhishan_Modules/bhishan/help/' + q +'.html'
    if os.path.isfile(path_html):
        mylst = open(home + path_html).readlines()
        mymd = '\n'.join(mylst)
        display(HTML(mymd))
        return

    # markdown
    path_md = 'Dropbox/a00_Bhishan_Modules/bhishan/help/' + q +'.md'
    if os.path.isfile(path_md):
        mylst = open(home + path_md).readlines()
        mymd = '\n'.join(mylst)
        display(Markdown(mymd))