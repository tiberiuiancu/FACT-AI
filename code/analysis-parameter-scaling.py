"""
plots the results of `make parameter-scaling`
"""

import pandas as pd
import matplotlib.pyplot as plt

FONTSIZE = 25
TICKS_SIZE = 15

df = pd.read_csv('output/parameter-scaling.csv')
df = df[['model', 'hid_dim', 'sp_mean', 'eo_mean']]
df['unfairness'] = df['sp_mean'] + df['eo_mean']

plt.figure(figsize=(8, 6))

for model in df['model'].unique():
    plt.plot(df[df['model'] == model]['hid_dim'], df[df['model'] == model]['unfairness'], label=model, marker='o')

plt.rc('font', size=FONTSIZE)
plt.xscale('log')
plt.grid()
plt.ylim(5, 36)
plt.xticks(df['hid_dim'], labels=df['hid_dim'], fontsize=TICKS_SIZE)
plt.yticks(fontsize=TICKS_SIZE)
plt.title('Unfairness vs hidden dimensionality', fontsize=FONTSIZE, pad=20)
plt.xlabel('Hidden Dimension Size', fontsize=FONTSIZE)
plt.ylabel('Unfairness (SP + EO)', fontsize=FONTSIZE)
plt.legend()
plt.savefig('output/parameter-scaling.pdf')
