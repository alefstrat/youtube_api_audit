import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict
import numpy as np


def jaccard_index(set1, set2):
    overlap = len(set1.intersection(set2))
    total_n = len(set1.union(set2))
    return overlap/total_n


topics = ['blm', 'brexit', 'capriot', 'grammys', 'higgs', 'worldcup']

path = "/data/"

plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True, sharex=True)

hor_idx = 0
ver_idx = 0

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

    vid_ids = OrderedDict(sorted(vid_ids.items()))
    diff_first = []
    df_setdiffs = []
    diff_previous = []
    dp_setdiffs = []        
    
    first_date = list(vid_ids.keys())[0]
    previous_date = list(vid_ids.keys())[0]

    for date in vid_ids:
        diff_first.append(jaccard_index(vid_ids[date], vid_ids[first_date]))
        df_setdiffs.append([len(vid_ids[first_date].difference(vid_ids[date]))/len(vid_ids[date].union(vid_ids[first_date])),
                            len(vid_ids[date].difference(vid_ids[first_date]))/len(vid_ids[date].union(vid_ids[first_date]))])
        diff_previous.append(jaccard_index(vid_ids[date], vid_ids[previous_date]))
        dp_setdiffs.append([len(vid_ids[previous_date].difference(vid_ids[date]))/len(vid_ids[date].union(vid_ids[previous_date])),
                            len(vid_ids[date].difference(vid_ids[previous_date]))/len(vid_ids[date].union(vid_ids[previous_date]))])

        
        previous_date = date

    # try:
    ax = axs[hor_idx][ver_idx]

    xrange = np.array([(date - first_date).days for date in vid_ids.keys()])

    ax.plot(xrange, diff_previous, '^', label=r"$\Delta$previous", color='tab:blue')
    ax.plot(xrange, diff_first, 'o', label=r"$\Delta$first", color='tab:orange')  # , linestyle=(0, (5, 10)))
    ax.errorbar(x=xrange-0.1, y=diff_previous, yerr=np.array(dp_setdiffs).T, capsize=3, alpha=0.5, color='tab:blue')
    ax.errorbar(x=xrange+0.1, y=diff_first, yerr=np.array(df_setdiffs).T, capsize=3, alpha=0.5, color='tab:orange')
    if ver_idx == 0:
        ax.set_ylabel('Rolling Jaccard similarity', fontsize=12)
    if hor_idx == axs.shape[0]-1:
        ax.set_xlabel('Collection day', fontsize=12)
    ax.set_title(topic)

    ax.legend(frameon=True, framealpha=0.2, facecolor='grey')

    hor_idx += 1
    if hor_idx == axs.shape[0]:
        hor_idx = 0
        ver_idx += 1

plt.savefig('./figures/video_jaccard.pdf', bbox_inches='tight', dpi=100)
