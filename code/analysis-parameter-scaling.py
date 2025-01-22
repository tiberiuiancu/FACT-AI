import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output/parameter-scaling.csv')
df = df[['model', 'hid_dim', 'sp_mean', 'eo_mean']]
df['unfairness'] = df['sp_mean'] + df['eo_mean']

for model in df['model'].unique():
    plt.plot(df[df['model'] == model]['hid_dim'], df[df['model'] == model]['unfairness'], label=model)

plt.savefig('output/parameter-scaling.png')
