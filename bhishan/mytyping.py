__version__ = 1.0
__author__ = 'Bhishan Poudel'


# Imports
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame, Series
from pandas.io.formats.style import Styler

# from mytyping import (IN, SN, SI, SIN, TL, LD, DS, DSt, NUM, NUMN,
#                         AD, AS, DN, ARR, ARRN, SARR, LIMIT, LIMITN)

IN = Union[int,None]
SN = Union[str,None]
SI = Union[str,int]
SIN = Union[str,int,None]

TL = Union[Tuple,List]
LD = Union[List,Dict]

DS = Union[DataFrame,Series]
DSt = Union[DataFrame,Styler]

NUM = Union[int,float]
NUMN = Union[int,float,None]

AD = Union[np.array,DataFrame]
AS = Union[np.array,Series]

DN = Union[DataFrame,None]

ARR = Union[Tuple,List,np.array,Series]
ARRN = Union[Tuple,List,np.array,Series,None]

SARR = Union[str,Tuple,List,np.array,Series]
SARRN = Union[str,Tuple,List,np.array,Series]

LIMIT = Union[Tuple[int,int],Tuple[float,float]]
LIMITN = Union[Tuple[int,int],Tuple[float,float],None]