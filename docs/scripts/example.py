import numpy as np
import pandas as pd
pd.options.display.max_colwidth=999
from bp.ds_json import MyJson

df = pd.DataFrame({'id': [0],
                        'payload': [\"""{"analytics": {"device": "Desktop",
                                                        "email_open_rate_pct": 14.0},
                                        "industry": "Construction",
                                        "time_in_product_mins": 62.45}\"""]
                        })
mj = MyJson()
df_out  = mj.parse_json_col(df,'payload')
print(df)
print('\n\n')
print('='*40)
print(df_out)
