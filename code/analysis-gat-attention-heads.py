"""
plots the results of `make gat-attention-heads`
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output/gat_attention_heads.csv')
df = df[['model', 'att_heads', 'sp_mean', 'eo_mean']]
df['unfairness'] = df['sp_mean'] + df['eo_mean']

plt.bar(df['att_heads'], df['unfairness'], color=['b', 'r'])

plt.title('GAT performance with different number of attention heads')
plt.ylabel('Unfairness')
plt.savefig('output/gat_attention_heads.png')
