# Module Installation
- Make sure you are in the root directory where there is setup.py file.
- Make sure to use -e option for the future edit of the module.

```python
pip install -e .
```

# Usage
```python
import numpy as np
import pandas as pd
import seaborn as sns
from bhishan import bp

%load_ext autoreload
%autoreload 2

df = sns.load_dataset('titanic')
df.bp.plot_num('age')
```

# Install module in gcolab
```python
%%capture
# capture will not print in notebook

import os
import sys
ENV_COLAB = 'google.colab' in sys.modules

if ENV_COLAB:
    ### mount google drive
    from google.colab import drive
    drive.mount('/content/drive')

    ### load the data dir
    dat_dir = 'drive/My Drive/Colab Notebooks/data/'
    sys.path.append(dat_dir)

    ### Image dir
    img_dir = 'drive/My Drive/Colab Notebooks/images/'
    if not os.path.isdir(img_dir): os.makedirs(img_dir)
    sys.path.append(img_dir)

    ### Output dir
    out_dir = 'drive/My Drive/Colab Notebooks/outputs/'
    if not os.path.isdir(out_dir): os.makedirs(out_dir)
    sys.path.append(out_dir)

    ### Also install my custom module
    module_dir = 'drive/My Drive/Colab Notebooks/Bhishan_Modules/'
    sys.path.append(module_dir)
    !cd drive/My Drive/Colab Notebooks/Bhishan_Modules/
    !pip install -e .
    !cd -
```
