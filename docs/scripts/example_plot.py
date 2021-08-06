import numpy as np
import pandas as pd
pd.options.display.max_colwidth=999
import matplotlib.pyplot as plt

from bp.plot_ds import plot_distribution

df = pd.DataFrame({'age': np.random.randint(20, 80, 100),
                    'salary': np.random.randint(50_000, 100_000, 100)})
print('\n')
print('Original Dataframe')
print('='*60)
print(df.head())


print('\n\n')
print('='*60)
print("Plotting histogram")

ax = plot_distribution(df, 'age')
plt.show()
