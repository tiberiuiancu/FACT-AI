import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output/gat_node_selection_model.csv')
df = df[['model', 'mode', 'sp_mean', 'eo_mean']]
df['unfairness'] = df['sp_mean'] + df['eo_mean']

plt.bar(df['mode'], df['unfairness'], color=['b', 'r'])

plt.title('GAT Node Selection Mode: uncertainty vs degree')
plt.ylabel('Unfairness')
plt.savefig('output/surrogate.png')
