__version__ = 1.0
__author__ = 'Bhishan Poudel'


# Imports
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
from typing import Optional, Sequence, Type, TypeVar
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame, Series
from pandas.io.formats.style import Styler

# from mytyping import (IN, SN, SI, SIN, TL, LD, DS, DSt, NUM, NUMN,
#                         AD, AS, DN, ARR, ARRN, SARR, LIMIT, LIMITN)

IN = Union[int,None]
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

# list and tuples
Lii = List[Union[int, int]]
Lff = List[Union[float, float]]
Lsi = List[Union[str, int]]
Lss = List[Union[str, str]]

Tii = Tuple[Union[int, int]]
Tff = Tuple[Union[float, float]]
Tsi = Tuple[Union[str, int]]
Tss = Tuple[Union[str, str]]

LTii = Union[Lii,Tii]
LTff = Union[Lff,Tff]
LTsi = Union[Lsi,Tsi]
LTss = Union[Lss,Tss]

# limits
LIMIT = Union[Tii,Lii]
LIMITN = Union[Tii,Lii,None]
