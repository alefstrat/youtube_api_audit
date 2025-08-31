import json
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datetime import datetime
from collections import OrderedDict, Counter
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates


def jaccard_index(set1, set2):
    overlap = len(set1.intersection(set2))
    total_n = len(set1.union(set2))
    # total_n = np.min([len(set1), len(set2)])
    return overlap/total_n


topics = ['blm', 'brexit', 'capriot', 'grammys', 'higgs', 'worldcup']

path = "/data/"

plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True, sharex=True)
# cmap = ListedColormap('turbo')

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
    curr_key = list(vid_ids.keys())[0]

    comps = 1
    # print('-'*15+'\n')

    # pd_dict = {'id': [], 't1': [], 't2': [], 'jaccard': []}
    pd_dict = {'id': [], 't-1': [], 't': [], 'jac_prev': [], 'jac_start': []}

    for date in vid_ids:
        if date == curr_key:  # log starting set and ignore first key
            start_set = vid_ids[curr_key]
            filename_start = curr_key.strftime("%b_%d").lower() + "_details.ndjson"
            continue

        common_ids = vid_ids[date].intersection(vid_ids[curr_key])
        common_start = vid_ids[date].intersection(start_set)
        
        filename1 = curr_key.strftime("%b_%d").lower() + "_details.ndjson"
        filename2 = date.strftime("%b_%d").lower() + '_details.ndjson'

        totals_f1 = set()
        totals_f2 = set()
        totals_startcomp = set()
        totals_startstr = set()

        with open(os.path.join(topicpath, filename1), 'r') as f:
            for line in f:
                raw = json.loads(line)
                if raw['id'] in common_ids:
                    totals_f1.add(raw['id'])
        
        with open(os.path.join(topicpath, filename2), 'r') as f:
            for line in f:
                raw = json.loads(line)
                if raw['id'] in common_ids:
                    totals_f2.add(raw['id'])
                if raw['id'] in common_start:
                    totals_startcomp.add(raw['id'])

        with open(os.path.join(topicpath, filename_start), 'r') as f:
            for line in f:
                raw = json.loads(line)
                if raw['id'] in common_start:
                    totals_startstr.add(raw['id'])

        print(f"{topic}: {len(common_start.union(totals_f2))}, Comp ID: {comps}")

        # print(f"\n{topic}: Comparing {curr_key} and {date}. Totals: {len(totals_f1)}/{len(common_ids)} and {len(totals_f2)}/{len(common_ids)}")
        # print(jaccard_index(totals_f1, totals_f2))
        pd_dict['id'].append(comps)
        pd_dict['t-1'].append(len(totals_f1)/len(common_ids))
        pd_dict['t'].append(len(totals_f2)/len(common_ids))
        pd_dict['jac_prev'].append(jaccard_index(totals_f1, totals_f2))
        pd_dict['jac_start'].append(jaccard_index(totals_startcomp, totals_startstr))

        curr_key = date

        comps += 1
    
    pd_df = pd.DataFrame.from_dict(pd_dict)

    ax = axs[hor_idx][ver_idx]
    parallel_coordinates(pd_df, 'id', ax=ax, colormap='rainbow')

    if ver_idx == 0:
        ax.set_ylabel('API Coverage', fontsize=12)
    # if hor_idx == axs.shape[0]-1:
    #     ax.set_xlabel('Collection day', fontsize=12)
    ax.set_title(topic)
    ax.grid(alpha=0.5, color='grey', linestyle='--')
    ax.get_legend().remove()

    # ax.legend(frameon=True, framealpha=0.1, facecolor='grey')

    hor_idx += 1
    if hor_idx == axs.shape[0]:
        hor_idx = 0
        ver_idx += 1

fig.legend(labels=list(pd_df['id']), loc='center right', title='Comparison ID')

plt.savefig('./figures/details_parallelplot.pdf')
