"""
plots the results of `make gat-attention-heads`
"""

import pandas as pd
import matplotlib.pyplot as plt

TICK_SIZE = 15
FONT_SIZE = 25

df = pd.read_csv('output/gat_attention_heads.csv')
df = df[['model', 'att_heads', 'sp_mean', 'eo_mean']]
df['unfairness'] = df['sp_mean'] + df['eo_mean']

plt.figure(figsize=(8, 6))
plt.plot(df['att_heads'], df['unfairness'], marker='o')

plt.rc('font', size=FONT_SIZE)
plt.xscale('log')
plt.grid()
plt.xticks(df['att_heads'], labels=df['att_heads'], fontsize=TICK_SIZE)
plt.minorticks_off()
plt.yticks(fontsize=TICK_SIZE)

plt.title('Unfairness vs # of attention heads', pad=20, fontsize=FONT_SIZE)
plt.xlabel('Number of attention heads', fontsize=FONT_SIZE)
plt.ylabel('Unfairness (SP + EO)', fontsize=FONT_SIZE)
plt.savefig('output/gat_attention_heads.pdf')
