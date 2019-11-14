import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()


diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)



os.makedirs('viz-plots/seaborn_heatmap', exist_ok=True)

sns.set()

fig, ax = plt.subplots(figsize=(14,14))
sns.heatmap(diabetes_df.corr(), annot=True, ax=ax, cmap='ocean', fmt='.2f', annot_kws={"size": 15}, linewidths=.05)
ax.set_xticklabels(diabetes_df.columns, rotation=45)
ax.set_yticklabels(diabetes_df.columns, rotation=45)
fig.subplots_adjust(top=.75)
plt.savefig('viz-plots/seaborn_heatmap/diabetes_heatmap.png')
# plt.tight_layout()
plt.close()

diabetes_df.hist(bins=14, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=6, ylabelsize=6, grid=False)
# plt.tight_layout(rect=(0, 0, 1.2, 1.2))
plt.savefig('viz-plots/seaborn_heatmap/diabetes_histogram.png')