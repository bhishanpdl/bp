# We have a class called MyJson in bp_json.py file of our module.
# We will run that function here.

import numpy as np
import pandas as pd
import bp
from bp.ds_json import MyJson

df = pd.DataFrame({'id': [0],
        'payload': ["""{"analytics": {"device": "Desktop",
                                        "email_open_rate_pct": 14.0},
                        "industry": "Construction",
                        "time_in_product_mins": 62.45}"""]
        })

# we can see the json key analytics has multiple values.
# we will expand all values and put them in a new column.

mj = MyJson()
df_out = mj.parse_json_col(df,'payload')

print(df_out)