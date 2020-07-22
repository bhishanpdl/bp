#!/Users/poudel/opt/miniconda3/envs/dataSc/bin/python
# -*- coding: utf-8 -*-#
"""
* File Name : using_pandas_api_extension.py

* Purpose : Example to use panda api extension

* Creation Date : Jun 02, 2020 Tue

* Last Modified : Tue Jun  2 13:59:22 2020

* Created By :  Bhishan Poudel

"""
# Imports
import bhishan
import sidetable

import numpy as np
import pandas as pd
import seaborn as sns
import calendar


df = sns.load_dataset('titanic')
print(df.columns)
print(df.head())

# freq
#print(df.bp.freq(['class']))

# plot
#df.bp.plot_num('age',print_=True)
#df.bp.plot_cat('pclass')
#df.bp.plot_num_num('pclass','age',save=True)
#df.bp.plot_num_cat('pclass','sex',save=True,show=True)
#df.bp.plot_cat_num('pclass','age',save=True,show=True)
#df.bp.plot_cat_cat('pclass','survived',save=True,show=True)

#=============== Datetime dataframes =================
ts = pd.date_range(start='1/1/2018',
                    end='2/1/2019',freq='H')
target = np.random.choice([0,1],size=len(ts))
df_ts = pd.DataFrame({'date': ts, 'target': target})
#df_ts.bp.plot_daily_cat('date','target',save=True,show=True)

# df.bp.plot_boxplot_cats_num(['pclass','sex'],'age',show=True,save=True)

# df.bp.plot_corrplot_with_pearsonr(['age','fare'],save=True)

# df.bp.plot_count_cat('pclass')

# df.bp.plot_corr(cols=['pclass','age'],show=True,save=True)

# df.bp.plot_cat_cat2('pclass','survived',show=True,save=True)

# df.bp.plot_num_cat('age','survived',show=True,save=True)

#==================================================
# df.bp.get_column_descriptions(print_=True)
df.bp.get_most_correlated_features(print_=True)
