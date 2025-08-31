import pandas as pd
import json
from datetime import datetime, timedelta
import os
import statsmodels.formula.api as smf
import isodate
import warnings
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.formula.api as smf
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler


class CLogLog(stats.rv_continuous):
    def _ppf(self, q):
        return np.log(-np.log(1 - q))

    def _cdf(self, x):
        return 1 - np.exp(-np.exp(x))


warnings.filterwarnings('ignore')

with open('./queries.json', 'r') as f:
    queries = json.load(f)

topics = ['blm', 'brexit', 'capriot', 'grammys', 'higgs', 'worldcup']

path = "/data/aefstra_data/yt_audit_data/"
errorcount = 0

for topic in topics:
    topicpath = f"{path}/{topic}/"
    print(f"Getting {topic.upper()}...")

    pub_after = datetime.fromisoformat(queries[topic]['focal_date']) + timedelta(days=14)

    ids = {}
    vid_dets = {}
    chan_dets = {}

    for file in os.listdir(topicpath):
        if file.endswith("_videos.ndjson"):
            
            date = file.strip("_videos.ndjson")
            date = datetime.strptime(date, "%b_%d").replace(year=2025)

            with open(os.path.join(topicpath, file), 'r') as f:
                for line in f:
                    raw = json.loads(line)
                    cur_id = raw['id']['videoId']
                    if cur_id in ids:
                        ids[cur_id].add(date)
                    else:
                        ids[cur_id] = {date}

        elif file.endswith("_details.ndjson"):
            with open(os.path.join(topicpath, file), 'r') as f:
                for line in f:
                    raw = json.loads(line)
                    cur_id = raw['id']
                    vid_dets[cur_id] = {'channel': raw['snippet']['channelId'],
                                        # 'category': raw.get('categoryId'),
                                        'duration': isodate.parse_duration(raw['contentDetails']['duration']).total_seconds(),
                                        'quality': raw['contentDetails']['definition'],
                                        'views': raw['statistics']['viewCount'],
                                        'likes': raw['statistics'].get('likeCount'),
                                        'comments': raw['statistics'].get('commentCount')
                                        # 'favorites': raw['statistics']['favoriteCount']}
                                        }
                    
        elif file.endswith("_channels.ndjson"):
            with open(os.path.join(topicpath, file), 'r') as f:
                for line in f:
                    raw = json.loads(line)
                    cur_id = raw['id']
                    try:
                        chan_dets[cur_id] = {'channel_age': (pub_after - datetime.fromisoformat(raw['snippet']['publishedAt'])).days,
                                            'channel_views': raw['statistics']['viewCount'],
                                            'channel_subs': raw['statistics']['subscriberCount'],
                                            'channel_numvids': raw['statistics']['videoCount']}
                    except KeyError:
                        errorcount += 1

    vid_dets = pd.DataFrame.from_dict(vid_dets).T
    vid_dets = vid_dets.apply(pd.to_numeric, errors='ignore')
    vid_dets.reset_index(inplace=True)
    vid_dets.rename(columns={"index": "id"}, inplace=True)
    vid_dets[vid_dets.select_dtypes(include=np.number).columns] = vid_dets.select_dtypes(include=np.number).apply(np.log1p)

    chan_dets = pd.DataFrame.from_dict(chan_dets).T
    chan_dets = chan_dets.apply(pd.to_numeric, errors='ignore')
    chan_dets.reset_index(inplace=True)
    chan_dets.rename(columns={"index": "channel"}, inplace=True)
    chan_dets[chan_dets.select_dtypes(include=np.number).columns] = chan_dets.select_dtypes(include=np.number).apply(np.log1p)
        
    for key in ids:
        ids[key] = len(ids[key])

    freq_df = pd.DataFrame(ids.items(), columns=['id', 'freq']).sort_values(by='freq', ascending=False)

    reg_df = vid_dets.merge(chan_dets, on='channel')
    reg_df = reg_df.merge(freq_df, on='id')
    reg_df['quality'] = pd.Categorical(reg_df['quality'])
    reg_df['topic'] = topic

    try:
        full_df = pd.concat([full_df, reg_df], axis=0)
    except NameError:
        full_df = reg_df


# add ordinal bins to the df
full_df['topic'] = pd.Categorical(full_df['topic'])
bins = [0, 5, 10, 15, 16]
labels = ['1-5', '6-10', '11-15', '16']

full_df['freq_cat'] = pd.cut(full_df['freq'], bins=bins, labels=labels, right=True)
full_df['freq_cat'] = pd.Categorical(full_df['freq_cat'])

full_std = full_df.copy()

continuous_vars = ['duration', 'views', 'likes', 'comments', 'channel_age', 'channel_views', 'channel_subs', 'channel_numvids']

scaler = StandardScaler()
full_std[continuous_vars] = scaler.fit_transform(full_std[continuous_vars])

# OLS model
model = smf.ols(formula="freq ~ duration + views + likes + comments + channel_age + channel_views + channel_subs + channel_numvids + C(quality) + C(topic)",
                    data=full_std).fit(cov_type='hc3')
# print(model.summary())

# binned ordinal model
modf_logit = OrderedModel.from_formula("freq_cat ~ duration + views + likes + comments + channel_age + channel_views + channel_subs + channel_numvids + C(quality) + C(topic)",
                    data=full_std, distr='logit').fit(method='bfgs')
print(modf_logit.summary())
modf_stat = 2 * (modf_logit.llf - modf_logit.llnull)
modf_p = stats.chi2.sf(modf_stat, df=modf_logit.df_model)

# scaling to get distribution parameters
y = np.array(full_std['freq'])
a, b = stats.expon.fit(y)

# full ordinal model
cloglog = CLogLog()
modf_fullordinal = OrderedModel.from_formula("freq ~ duration + views + likes + comments + channel_age + channel_views + channel_subs + channel_numvids + C(quality) + C(topic)",
                    data=full_std, distr=cloglog).fit(method='bfgs')
full_stat = 2 * (modf_fullordinal.llf - modf_fullordinal.llnull)
full_p = stats.chi2.sf(full_stat, df=modf_fullordinal.df_model)

with open("./results/regression.txt", "w+") as fw:
    fw.write("CONTINUOUS REGRESSION RESULTS\n\n")
    fw.write(f"{model.summary().as_text()}\n\n\n")
    fw.write("ORDINAL -- BINNED (Link function: Logit)\n\n")
    fw.write(f"{modf_logit.summary().as_text()}\n")
    fw.write(f"Fit: {modf_stat}, {modf_p}\nMcFadden (pseudo) R2: {modf_logit.prsquared}\n\n\n")
    fw.write("ORDINAL -- ALL CATS (Link function: CLogLog)\n\n")
    fw.write(f"{modf_fullordinal.summary().as_text()}\n")
    fw.write(f"Fit: {full_stat}, {full_p}\nMcFadden (pseudo) R2: {modf_fullordinal.prsquared}\n\n\n")

print('end')
