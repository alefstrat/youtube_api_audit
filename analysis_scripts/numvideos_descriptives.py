import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

topics = ['blm', 'brexit', 'capriot', 'grammys', 'higgs', 'worldcup']

path = "/data/"

df = {'topic': [], 'min': [], 'max': [], 'mean': [], 'sd': []}

for topic in topics:
    topicpath = f"{path}/{topic}/"

    vid_ids = {}

    for file in os.listdir(topicpath):
        if file.endswith("_videos.ndjson"):
            date = file.strip("_videos.ndjson")
            date = datetime.strptime(date, "%b_%d").replace(year=2025)

            vid_ids[date] = set()

            with open(os.path.join(topicpath, file), 'r') as f:
                for line in f:
                    raw = json.loads(line)
                    vid_ids[date].add(raw['id']['videoId'])

    numvids = [len(vid_ids[key]) for key in vid_ids]
    df['topic'].append(topic)
    df['min'].append(np.min(numvids))
    df['max'].append(np.max(numvids))
    df['mean'].append(np.mean(numvids))
    df['sd'].append(np.std(numvids))

df = pd.DataFrame.from_dict(df)  #, orient='index')
df.to_csv('./results/numvids_descriptives.csv', index=False)

print('end')
