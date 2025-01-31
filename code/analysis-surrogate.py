"""
plots the results of `make surrogate`
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output/surrogate.csv')
df = df[['model', 'surrogate', 'sp_mean', 'eo_mean']]
df['unfairness'] = df['sp_mean'] + df['eo_mean']
df['setting'] = 'Model ' + df['model'] + '\nSurrogate ' + df['surrogate']

plt.bar(df['setting'], df['unfairness'], color=['b', 'r', 'g', 'y'])

plt.ylabel('Unfairness')
plt.savefig('output/surrogate.png')
