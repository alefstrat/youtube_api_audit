import pandas as pd
import json
from datetime import datetime, timedelta
import os
import isodate
import warnings
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_poisson_deviance
import lightgbm as lgb
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_poisson_deviance


warnings.filterwarnings('ignore')

with open('./queries.json', 'r') as f:
    queries = json.load(f)

topics = ['blm', 'brexit', 'capriot', 'grammys', 'higgs', 'worldcup']

path = "/data/"
errorcount = 0

# prepare dataframe
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
                                        'duration': isodate.parse_duration(raw['contentDetails']['duration']).total_seconds(),
                                        'quality': raw['contentDetails']['definition'],
                                        'views': raw['statistics']['viewCount'],
                                        'likes': raw['statistics'].get('likeCount'),
                                        'comments': raw['statistics'].get('commentCount')
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

full_df['topic'] = pd.Categorical(full_df['topic'])

full_df.drop(columns=['id', 'channel'], inplace=True)

y = full_df["freq"]  # outcome vector
X = full_df.drop(columns=["freq"])  # predictor feature matrix

# one-hot encode categoricals
X = pd.get_dummies(X, drop_first=False)  # keep all dummies to see impact of every category

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM with poisson regression (i.e., count outcome)
model = lgb.LGBMRegressor(objective="poisson", random_state=42)
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

plt.style.use('seaborn-v0_8-whitegrid')

# bar plot for SHAP values; gives metrics for feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.savefig('./figures/shap_bar.png', dpi=100, bbox_inches='tight')
plt.close()

# beeswarm plot for SHAP value distributions; visualizes distribution of feature importance per data point
shap.summary_plot(shap_values, X_test)
plt.savefig('./figures/shap_beeswarm.png', dpi=100, bbox_inches='tight')
plt.close()

# heatmap plot
shap.plots.heatmap(shap_values, max_display=len(X.columns))
plt.savefig('./figures/shap_heatmap.png', dpi=100, bbox_inches='tight')
plt.close()

# model performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R2 score: {r2}")
