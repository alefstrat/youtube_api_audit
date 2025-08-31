import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import warnings
import seaborn as sns


def jaccard_index(set1, set2):
    overlap = len(set1.intersection(set2))
    total_n = len(set1.union(set2))
    try:
        return overlap/total_n
    except ZeroDivisionError:
        return 1.0


warnings.filterwarnings("ignore")

topics = ['blm', 'brexit', 'capriot', 'grammys', 'higgs', 'worldcup']
# topics = ['capriot']

path = "/data/"

plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 3, figsize=(12, 8))  #, sharey=False, sharex=False)

hor_idx = 0
ver_idx = 0

with open('queries.json', 'r') as f:
    query_info = json.load(f)

onetailed_span = 14

for topic in topics:
    topicpath = f"{path}/{topic}/"

    topic_dfs_daily = dict()

    min_date = datetime.fromisoformat(query_info[topic]['focal_date']) - timedelta(days=onetailed_span)
    max_date = datetime.fromisoformat(query_info[topic]['focal_date']) + timedelta(days=onetailed_span)

    for file in os.listdir(topicpath):
        if file.endswith("_videos.ndjson"):
            date = file.strip("_videos.ndjson")
            date = datetime.strptime(date, "%b_%d").replace(year=2025)

            topic_dfs_daily[date] = []
            with open(os.path.join(topicpath, file), 'r') as f:
                for line in f:
                    raw = json.loads(line)
                    topic_dfs_daily[date].append([raw['id']['videoId'], raw['snippet']['publishedAt']])

                # hacky way to force date boundaries
                topic_dfs_daily[date].append(['plch', min_date])
                topic_dfs_daily[date].append(['plch', max_date])

    topic_dfs_daily = OrderedDict(sorted(topic_dfs_daily.items()))

    for key in topic_dfs_daily:
        topic_dfs_daily[key] = pd.DataFrame(topic_dfs_daily[key], columns=['id', 'pubtime'])
        topic_dfs_daily[key]['pubtime'] = pd.to_datetime(topic_dfs_daily[key]['pubtime'])
        colname = key.strftime("%b_%d")
        # aggregate all video ids per hour they were posted
        topic_dfs_daily[key] = topic_dfs_daily[key].groupby(pd.Grouper(key='pubtime', freq='1D')).agg(idset=('id', lambda x: set(x)))
        topic_dfs_daily[key].rename(columns={'idset': colname}, inplace=True)
        # remove the artificial count on min and max that was introduced to force date boundaries (first and last indices)
        topic_dfs_daily[key][colname][0].remove('plch')
        topic_dfs_daily[key][colname].iloc[-1].remove('plch')

    # merge all dates to compare
    topic_dfs_daily = pd.concat(list(topic_dfs_daily.values()), axis=1).loc[:, ~pd.concat(list(topic_dfs_daily.values()), axis=1).columns.duplicated()]
    # get hourly jaccard similarity between first and last collection; write it at the end so we can perform count averaging
    jac_sims = [jaccard_index(first, last) for first, last in zip(topic_dfs_daily.iloc[:, 0], topic_dfs_daily.iloc[:, -1])]
    topic_dfs_daily = topic_dfs_daily.applymap(len)

    first_date = topic_dfs_daily.columns[0]
    last_date = topic_dfs_daily.columns[-1]

    topic_dfs_daily['avg_count'] = topic_dfs_daily.mean(axis=1, numeric_only=True)
    topic_dfs_daily['jaccard'] = jac_sims
    # topic_dfs_hourly = topic_dfs_hourly.loc[topic_dfs_hourly['avg_count'] > 0]

    ax = axs[hor_idx][ver_idx]

    ax.plot(topic_dfs_daily.index, topic_dfs_daily['avg_count'], color='black', label='Average frequency')
    ax.plot(topic_dfs_daily.index, topic_dfs_daily[first_date], color='tab:orange', label='Frequency $T_{1}$')
    ax.plot(topic_dfs_daily.index, topic_dfs_daily[last_date], color='tab:blue', label='Frequency $T_{L}$')
    ax.tick_params(axis='x', labelrotation = 90)

    foc_date = pd.to_datetime(query_info[topic]['focal_date'])
    ax.axvline(x=foc_date, linestyle='--', color='grey')

    ax2 = ax.twinx()
    ax2.plot(topic_dfs_daily.index, topic_dfs_daily['jaccard'], color='red', label='Jaccard similarity')
    ax2.set_yticks(np.linspace(0, 1, 11))
    ax2.grid(False)

    # sns.kdeplot(topic_dfs_hourly, y='jaccard', x='pubtime', fill=False, bw_adjust=0.01, ax=ax2, label='Jaccard similarity', color='red', common_norm=False, legend=False)
    # ax2.set_ylim(0, 1)
    # ax.get_legend().remove()

    if ver_idx == 0:
        ax.set_ylabel('Frequency', fontsize=12)
    elif ver_idx == 2:
        ax2.set_ylabel('Jaccard Similarity', fontsize=12)
    if hor_idx == axs.shape[0]-1:
        ax.set_xlabel('Date', fontsize=12)
    ax.set_title(topic)

    if ver_idx != 2:
        ax2.set_yticklabels([])

    # ax.legend(frameon=True, framealpha=0.2, facecolor='grey')

    hor_idx += 1
    if hor_idx == axs.shape[0]:
        hor_idx = 0
        ver_idx += 1

handles, labels = [], []

h, l = ax.get_legend_handles_labels()  # Get handles and labels from ax
handles.extend(h)
labels.extend(l)

h2, l2 = ax2.get_legend_handles_labels()  # Get handles and labels from ax2
handles.extend(h2)
labels.extend(l2)

fig.legend(handles, labels, bbox_to_anchor=(1.17, 0.55), fontsize=11)

plt.tight_layout()
plt.savefig('./figures/daily_breakdown.pdf', bbox_inches='tight', dpi=100)
