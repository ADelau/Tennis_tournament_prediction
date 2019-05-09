import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

raw_df = pd.read_csv('correlation.csv', index_col=False, names=['player', 'surface', 'ratio'])


surfaces = ['Grass', 'Hard', 'Carpet', 'Clay']

# Trim NaN and None surfaces
raw_df = raw_df.dropna()
raw_df = raw_df[raw_df.surface != 'None']


dic = {s: {p: np.nan for p in raw_df.player.unique()} for s in surfaces}

for _, entry in raw_df.iterrows():
    dic[entry['surface']][entry['player']] = entry['ratio']


df_with_carpet = pd.DataFrame(dic).dropna()
corr = df_with_carpet.corr()
print(corr)

sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)


df_without_carpet = pd.DataFrame(dic).drop('Carpet', axis=1).dropna()
corr = df_without_carpet.corr()
print(corr)

sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)
plt.savefig("figures/corr.pdf")
