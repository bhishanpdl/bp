__version__ = 1.0
__author__ = 'Bhishan Poudel'

__doc__ = """
Utility function for json.

"""
import numpy as np
import pandas as pd
import json

class MyJson(object):
    def __init__(self):
        pass

    def parse_json_col(self, df,json_col):
        """Explode the json column and attach to original dataframe.

        Parameters
        -----------
        df: pandas.DataFrame
            input dataframe

        json_col: string
            Column name of dataframe which contains json objects.

        Example:
        --------
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
        ans = mj.parse_json_col(df,'payload')
        """
        # give increasing index to combine later
        df = df.reset_index()

        df_json = df[json_col].apply(json.loads).apply(pd.json_normalize)
        df_json = pd.concat(df_json.to_numpy())
        df_json.index = range(len(df_json))

        df_no_json = df.drop(json_col,axis=1)
        cols = df_no_json.columns.tolist() + df_json.columns.tolist()

        df_combined = pd.concat([df_no_json, df_json], axis=1, ignore_index=False)
        df_combined.columns = cols

        # retrieve the original index
        df_combined.set_index('index',inplace=True)
        return df_combined
