# We have a class called MyJson in bp_json.py file of our module.
# We will run that function here.
import numpy as np
import pandas as pd

try:
    import bp
    from bp.ds_json import MyJson
except:
    import sys
    sys.path.append("/Users/poudel/Dropbox/a00_Bhishan_Modules/")
    sys.path.append("/Users/poudel/Dropbox/a00_Bhishan_Modules/bhishan")
    from bhishan import bp
    from ds_json import MyJson

ifile = "data/my_data.json"
df = pd.read_json(ifile)
df['payload'] = df['payload'].astype(str).str.replace("'",'"')

# we can see the json key analytics has multiple values.
# we will expand all values and put them in a new column.

mj = MyJson()
df_out = mj.parse_json_col(df,'payload')

print(df_out)