__version__ = 1.0
__author__ = 'Bhishan Poudel'


# type hints
from typing import List,Tuple,Dict,Any,Callable,Iterable,Union
from typing import Optional, Sequence, Type, TypeVar
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame, Series
from pandas.io.formats.style import Styler
# try:
#     from .mytyping import (IN, SN, SI, SIN, TL, LD, TLN, LDN,
#     DS, DSt, NUM, NUMN, AD, AS, DN,
#     ARR, ARRN, SARR, SARRN, LIMIT, LIMITN,
#     LTii,LTss,LTff,LTsi,
#     )
# except:
#     from mytyping import (IN, SN, SI, SIN, TL, LD, TLN, LDN,
#     DS, DSt, NUM, NUMN, AD, AS, DN,
#     ARR, ARRN, SARR, SARRN, LIMIT, LIMITN,
#     LTii,LTss,LTff,LTsi,
#     )

IN = Union[int,None]
SN = Union[str,None] # eg. color
SI = Union[str,int]
SIN = Union[str,int,None]

TL = Union[Tuple,List]
LD = Union[List,Dict]
TLN = Union[Tuple,List,None]
LDN = Union[List,Dict,None]

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

# list and tuples: integer, float, string
Lii = List[Union[int, int]]
Lff = List[Union[float, float]]
Lss = List[Union[str, str]]
Lsi = List[Union[str, int]]


Tii = Tuple[Union[int, int]]
Tff = Tuple[Union[float, float]]
Tss = Tuple[Union[str, str]]
Tsi = Tuple[Union[str, int]]

LTii = Union[Lii,Tii]
LTff = Union[Lff,Tff]
LTss = Union[Lss,Tss]
LTsi = Union[Lsi,Tsi]

LTiiN = Union[Lii,Tii,None]
LTffN = Union[Lff,Tff,None]
LTssN = Union[Lss,Tss,None]
LTsiN = Union[Lsi,Tsi,None]

# limits
LIMIT = Union[Tii,Lii]
LIMITN = Union[Tii,Lii,None]
