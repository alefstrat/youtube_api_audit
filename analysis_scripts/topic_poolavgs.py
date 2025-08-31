import json
import os
import numpy as np
import pandas as pd
from scipy import stats

topics = ['blm', 'brexit', 'capriot', 'grammys', 'higgs', 'worldcup']

path = "/data/"

topic_stats = {}

for topic in topics:
    print(f"Running {topic}...")
    topicpath = f"{path}/{topic}/"

    totals = []
    returns = []

    for file in os.listdir(topicpath):
        if file.endswith("_metadata.ndjson"):
            with open(os.path.join(topicpath, file), 'r') as f:
                for line in f:
                    raw = json.loads(line)
                    numres = raw['pageInfo']['totalResults']
                    rets = raw['pageInfo']['resultsPerPage']
                    totals.append(numres)
                    returns.append(rets)

    print(f"{topic}: {stats.spearmanr(totals, returns)}")
    
    topic_stats[topic] = {}
    topic_stats[topic]['min'] = np.min(totals)
    topic_stats[topic]['max'] = np.max(totals)
    topic_stats[topic]['mean'] = np.mean(totals)
    topic_stats[topic]['mode'] = int(stats.mode(totals)[0])

summary = pd.DataFrame.from_dict(topic_stats, orient='index')
summary.to_csv('./results/topic_pools.csv')
