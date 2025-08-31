import json
import pandas as pd
import numpy as np
import os
from collections import OrderedDict, defaultdict, Counter
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt


def markov_transitions(data, order=1):
    # data should be the transition matrix as np.array
    transition_counts = defaultdict(Counter)

    for row in data:
        row = list(row)
        for i in range(len(row) - order):
            state = tuple(row[i:i+order])          # current state (order-k)
            next_state = row[i + order]            # state following the current state
            transition_counts[state][next_state] += 1

    # Normalize to probabilities
    transition_probs = {
        state: {k: v / sum(nexts.values()) for k, v in nexts.items()}
        for state, nexts in transition_counts.items()
    }

    return transition_probs


topics = ['blm', 'brexit', 'capriot', 'grammys', 'higgs', 'worldcup']

path = "/data/"

vid_ids = {}

for topic in topics:
    topicpath = f"{path}/{topic}/"

    for file in os.listdir(topicpath):
        if file.endswith("_videos.ndjson"):
            date = file.strip("_videos.ndjson")
            date = datetime.strptime(date, "%b_%d").replace(year=2025)

            if date not in vid_ids:
                vid_ids[date] = set()
            with open(os.path.join(topicpath, file), 'r') as f:
                for line in f:
                    raw = json.loads(line)
                    vid_ids[date].add(raw['id']['videoId'])

vid_ids = OrderedDict(sorted(vid_ids.items()))

full_set = set()

for date in vid_ids:
    full_set = full_set.union(vid_ids[date])

transition_matrix = []

with tqdm(total=len(full_set), desc="Building transition matrix") as pbar:
    for vid in full_set:
        transitions = []
        for date in vid_ids:
            if vid in vid_ids[date]:
                transitions.append(1)
            else:
                transitions.append(0)
        transition_matrix.append(transitions)
        pbar.update(1)

transition_matrix = np.array(transition_matrix)

prob_matrix = markov_transitions(transition_matrix, order=2)

plot_labels = []
states = ['P', 'A']
plot_map = {1: 'P', 0: 'A'}
plot_data = []

for key in prob_matrix:
    label = f"{plot_map[key[0]]}\u2192{plot_map[key[1]]}"
    plot_data.append([round(prob_matrix[key][1], 3), round(prob_matrix[key][0], 3)])
    plot_labels.append(label)

plot_data = np.array(plot_data).T

fig, ax = plt.subplots(figsize=(4,6))
ax.imshow(plot_data, cmap='coolwarm')

ax.set_xticks(range(len(plot_labels)), labels=plot_labels, ha="center", fontsize=12)
ax.set_yticks(range(len(states)), labels=states, fontsize=12)
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

for i in range(len(states)):
    for j in range(len(plot_labels)):
        text = ax.text(j, i, plot_data[i, j],
                       ha="center", va="center", fontsize=12)

plt.savefig('./figures/dropout_transitions.pdf', dpi=100, bbox_inches='tight')
