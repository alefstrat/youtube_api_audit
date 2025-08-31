import json
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import subprocess
import pandas as pd


def jaccard_index(set1, set2):
    overlap = len(set1.intersection(set2))
    total_n = len(set1.union(set2))
    try:
        return overlap/total_n
    except ZeroDivisionError:
        return None


def count_lines(filepath):
    result = subprocess.run(['wc', '-l', filepath], capture_output=True, text=True)
    if result.returncode == 0:
        return int(result.stdout.strip().split()[0])
    else:
        raise RuntimeError(f"Error counting lines: {result.stderr}")


with open('queries.json', 'r') as f:
    queries = json.load(f)

topics = ['blm', 'brexit', 'capriot', 'grammys', 'higgs', 'worldcup']
path = "/data/"

jaccard_dict = {'topic': [], 'sim_t_ns': [], 'sim_n_ns': [], 'sim_t_s': [], 'sim_n_s': []}

for topic in topics:
    topicpath = f"{path}/{topic}/"
    print(f"{topic}\n{'-'*15}")

    dates = []
    vid_ids = {}

    for file in os.listdir(topicpath):
        if file.endswith("_threads.ndjson"):
            date = file.removesuffix("_threads.ndjson")
            date = datetime.strptime(date, "%b_%d").replace(year=2025)
            dates.append(date)

    dates.sort()
    first_vids = dates[0].strftime("%b_%d").lower() + "_videos.ndjson"
    last_vids = dates[-1].strftime("%b_%d").lower() + "_videos.ndjson"

    first_set = set()
    last_set = set()

    with open(os.path.join(topicpath, first_vids), 'r') as f:
        for line in f:
            raw = json.loads(line)
            first_set.add(raw['id']['videoId'])
    
    with open(os.path.join(topicpath, last_vids), 'r') as f:
        for line in f:
            raw = json.loads(line)
            last_set.add(raw['id']['videoId'])

    shared_vids = first_set.intersection(last_set)

    first_threads = dates[0].strftime("%b_%d").lower() + "_threads.ndjson"
    first_comments = dates[0].strftime("%b_%d").lower() + "_comments.ndjson"
    last_threads = dates[-1].strftime("%b_%d").lower() + "_threads.ndjson"
    last_comments = dates[-1].strftime("%b_%d").lower() + "_comments.ndjson"

    first_toplevel = set()
    first_nested = set()
    last_toplevel = set()
    last_nested = set()

    # print(len(first_toplevel.union(first_nested)))

    first_toplevel_s = set()
    first_nested_s = set()
    last_toplevel_s = set()
    last_nested_s = set()

    datefilter = datetime.fromisoformat(queries[topic]["focal_date"]) + timedelta(days=21)  # 3 weeks after focal date to allow comment consolidation

    curr_file = os.path.join(topicpath, first_threads)

    with tqdm(total=count_lines(curr_file), desc=f"Processing {first_threads}") as pbar:
        with open(os.path.join(topicpath, first_threads), 'r') as f:
            for line in f:
                raw = json.loads(line)
                toplevel = raw['snippet']['topLevelComment']
                if datetime.fromisoformat(toplevel['snippet']['publishedAt']) < datefilter:
                    first_toplevel.add(raw['id'])
                    if raw['snippet']['videoId'] in shared_vids:
                        first_toplevel_s.add(raw['id'])

                if raw['snippet']['totalReplyCount'] > 0:
                    for reply in raw['replies']['comments']:
                        if datetime.fromisoformat(reply['snippet']['publishedAt']) < datefilter:
                            first_nested.add(reply['id'])
                            if raw['snippet']['videoId'] in shared_vids:
                                first_nested_s.add(reply['id'])

                pbar.update(1)

    curr_file = os.path.join(topicpath, first_comments)

    with tqdm(total=count_lines(curr_file), desc=f"Processing {first_comments}") as pbar:
        with open(os.path.join(topicpath, first_comments), 'r') as f:
            for line in f:
                raw = json.loads(line)
                if datetime.fromisoformat(raw['snippet']['publishedAt']) < datefilter:
                    first_nested.add(raw['id'])
                    if raw['snippet']['parentId'] in first_toplevel_s:
                        first_nested_s.add(raw['id'])
                pbar.update(1)

    curr_file = os.path.join(topicpath, last_threads)

    with tqdm(total=count_lines(curr_file), desc=f"Processing {last_threads}") as pbar:
        with open(os.path.join(topicpath, last_threads), 'r') as f:
            for line in f:
                raw = json.loads(line)
                toplevel = raw['snippet']['topLevelComment']
                if datetime.fromisoformat(toplevel['snippet']['publishedAt']) < datefilter:
                    last_toplevel.add(raw['id'])
                    if raw['snippet']['videoId'] in shared_vids:
                        last_toplevel_s.add(raw['id'])

                if raw['snippet']['totalReplyCount'] > 0:
                    try:
                        for reply in raw['replies']['comments']:
                            if datetime.fromisoformat(reply['snippet']['publishedAt']) < datefilter:
                                last_nested.add(reply['id'])
                                if raw['snippet']['videoId'] in shared_vids:
                                    last_nested_s.add(reply['id'])
                    except KeyError:
                        print(f"Error encountered at videoId {raw['id']}")


                pbar.update(1)

    curr_file = os.path.join(topicpath, last_comments)

    with tqdm(total=count_lines(curr_file), desc=f"Processing {last_comments}") as pbar:
        with open(os.path.join(topicpath, last_comments), 'r') as f:
            for line in f:
                raw = json.loads(line)
                if datetime.fromisoformat(raw['snippet']['publishedAt']) < datefilter:
                    last_nested.add(raw['id'])
                    if raw['snippet']['parentId'] in last_toplevel_s:
                        last_nested_s.add(raw['id'])
                pbar.update(1)

    jaccard_dict['topic'].append(topic)
    jaccard_dict['sim_t_ns'].append(jaccard_index(first_toplevel, last_toplevel))
    jaccard_dict['sim_n_ns'].append(jaccard_index(first_nested, last_nested))
    jaccard_dict['sim_t_s'].append(jaccard_index(first_toplevel_s, last_toplevel_s))
    jaccard_dict['sim_n_s'].append(jaccard_index(first_nested_s, last_nested_s))

    print('-'*15)

sim_df = pd.DataFrame.from_dict(jaccard_dict)
sim_df.to_csv('./results/comment_similarities.csv', index=False)
