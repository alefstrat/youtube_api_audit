import json
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from collections import OrderedDict
import os
import warnings

def jaccard_index(set1, set2):
    overlap = len(set1.intersection(set2))
    total_n = len(set1.union(set2))
    try:
        return overlap/total_n
    except ZeroDivisionError:
        return 1.0


warnings.filterwarnings("ignore")

topics = ['blm', 'brexit', 'capriot', 'grammys', 'higgs', 'worldcup']

path = "/data/"

with open('queries.json', 'r') as f:
    query_info = json.load(f)

onetailed_span = 14

descriptives = {'topic': [], 'mode': [], 'mean': [], 'min': [], 'max': [], '1qr': [], '3qr': [], 'std': [], 'rho': [], 'rho_sig': [], 'N_corr': []}

for topic in topics:
    topicpath = f"{path}/{topic}/"

    topic_dfs_hourly = dict()

    min_date = datetime.fromisoformat(query_info[topic]['focal_date']) - timedelta(days=onetailed_span)
    max_date = datetime.fromisoformat(query_info[topic]['focal_date']) + timedelta(days=onetailed_span)

    for file in os.listdir(topicpath):
        if file.endswith("_videos.ndjson"):
            date = file.strip("_videos.ndjson")
            date = datetime.strptime(date, "%b_%d").replace(year=2025)

            topic_dfs_hourly[date] = []
            with open(os.path.join(topicpath, file), 'r') as f:
                for line in f:
                    raw = json.loads(line)
                    topic_dfs_hourly[date].append([raw['id']['videoId'], raw['snippet']['publishedAt']])

                # hacky way to force date boundaries
                topic_dfs_hourly[date].append(['plch', min_date])
                topic_dfs_hourly[date].append(['plch', max_date])

    topic_dfs_hourly = OrderedDict(sorted(topic_dfs_hourly.items()))

    for key in topic_dfs_hourly:
        topic_dfs_hourly[key] = pd.DataFrame(topic_dfs_hourly[key], columns=['id', 'pubtime'])
        topic_dfs_hourly[key]['pubtime'] = pd.to_datetime(topic_dfs_hourly[key]['pubtime'])
        colname = key.strftime("%b_%d")
        # aggregate all video ids per hour they were posted
        topic_dfs_hourly[key] = topic_dfs_hourly[key].groupby(pd.Grouper(key='pubtime', freq='1h')).agg(idset=('id', lambda x: set(x)))
        topic_dfs_hourly[key].rename(columns={'idset': colname}, inplace=True)
        # remove the artificial count on min and max that was introduced to force date boundaries (first and last indices)
        topic_dfs_hourly[key][colname][0].remove('plch')
        topic_dfs_hourly[key][colname].iloc[-1].remove('plch')

    # merge all dates to compare
    topic_dfs_hourly = pd.concat(list(topic_dfs_hourly.values()), axis=1).loc[:, ~pd.concat(list(topic_dfs_hourly.values()), axis=1).columns.duplicated()]
    # get hourly jaccard similarity between first and last collection; write it at the end so we can perform count averaging
    jac_sim = [jaccard_index(first, last) for first, last in zip(topic_dfs_hourly.iloc[:, 0], topic_dfs_hourly.iloc[:, -1])]

    topic_dfs_hourly = topic_dfs_hourly.applymap(len)

    avg_count = list(topic_dfs_hourly.mean(axis=1, numeric_only=True))
    corr_df = pd.DataFrame([avg_count, jac_sim]).T 
    corr_df.columns = ['avg_count', 'jac_sim']
    corr_df = corr_df.loc[corr_df['avg_count'] > 0]

    first_date = topic_dfs_hourly.columns[0]
    last_date = topic_dfs_hourly.columns[-1]

    descs = topic_dfs_hourly.to_numpy().flatten()

    descriptives['topic'].append(topic)
    descriptives['mode'].append(int(stats.mode(descs)[0]))
    descriptives['mean'].append(np.mean(descs))
    descriptives['min'].append(descs.min())
    descriptives['max'].append(descs.max())
    descriptives['1qr'].append(np.percentile(descs, 25))
    descriptives['3qr'].append(np.percentile(descs, 75))
    descriptives['std'].append(np.std(descs))
    
    corr = stats.spearmanr(corr_df['jac_sim'], corr_df['avg_count'])
    descriptives['rho'].append(corr.statistic)
    descriptives['rho_sig'].append(corr.pvalue)
    descriptives['N_corr'].append(len(corr_df))

desc_df = pd.DataFrame.from_dict(descriptives)
desc_df.to_csv('./results/hourly_descriptive_stats.csv', index=False)
