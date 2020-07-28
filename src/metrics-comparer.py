import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

colors = ['#32cd32', '#1979d3']
sns.set_palette(sns.color_palette(colors))
sns.set_style("whitegrid", {'grid.linestyle': '--'})
sns.despine()

df_classic = pd.read_csv('../output/metrics.csv')
df_classic = df_classic.rename(columns={df_classic.columns[0]: 'Steps'})
# df_classic['Rolling Mean3'] = df_classic['Queued Time (AVG)'].rolling(4).mean()
df_classic['Mode'] = 'Classic'

df_snake = pd.read_csv('../output/metrics-snake.csv')
df_snake = df_snake.rename(columns={df_snake.columns[0]: 'Steps'})
# df_snake['Rolling Mean3'] = df_snake['Queued Time (AVG)'].rolling(4).mean()
df_snake['Mode'] = 'Snake'

df = df_classic.append(df_snake, ignore_index=True)
print(df)
# print(df.info())


ax = sns.lineplot(data=df, x='Customers in Queue', y='Average Time in Queue', hue='Mode', ci=None)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_xticks(ticks)
# ax.set_ylabel('Average Customers per Queue')
plt.tight_layout()
plt.savefig('../output/test')
plt.close()

print('In Store:', df['Average Time in Store'].mean())
print('In Queue:', df['Average Time in Queue'].mean())

ax = sns.distplot(df['Average Time in Store'])
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_xticks(ticks)
# ax.set_ylabel('Average Customers per Queue')
plt.tight_layout()
plt.savefig('../output/test')
plt.close()

ticks = np.arange(min(df['Steps']), max(df['Steps']) + 1, 795.0)

# fig = plt.figure(figsize=(10, 5))
# ax = fig.gca()
# sns.lineplot(data=df, x='Steps', y='Rolling Mean3', hue='Mode', ci=None, ax=ax)
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_xticks(ticks)
# ax.set_ylabel('Customers in Queue (Window size= 3)')
# plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(10, 5))
ax = fig.gca()
sns.lineplot(data=df, x='Steps', y='Customers in Queue', hue='Mode', ci=None, ax=ax)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticks(ticks)
ax.set_ylabel('Customers in Queue')
plt.tight_layout()
plt.savefig('../output/comparison-queued')
plt.close()
# plt.show()

ax = sns.lineplot(data=df, x='Steps', y='Average Customers per Queue', hue='Mode', ci=None)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticks(ticks)
ax.set_ylabel('Average Customers per Queue')
plt.tight_layout()
plt.savefig('../output/comparison-queued-avg')
plt.close()
# plt.show()

ax = sns.lineplot(data=df, x='Steps', y='Average Customers per Queue', hue='Mode', ci=None)
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticks(ticks)
ax.set_ylabel('Average Time in Queue')
plt.tight_layout()
plt.savefig('../output/comparison-queued-time-avg')
plt.close()
# plt.show()

ax = sns.lineplot(data=df, x='Steps', y='Average Time in Store', hue='Mode', ci=None)
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticks(ticks)
ax.set_ylabel('Average Time in Store')
plt.tight_layout()
plt.savefig('../output/comparison-total-time-avg')
plt.close()
# plt.show()
