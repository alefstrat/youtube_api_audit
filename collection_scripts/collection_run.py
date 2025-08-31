import youtube_api_calls as ytapi
import logging
import json
import os
from datetime import datetime, timedelta
from scheduler import schedule


dev_key = "YOUR_API_KEY"

with open('queries.json', 'r') as f:
        queries = json.load(f)

onetailed_span = 14  # determines how many days before and after focal date to collect; total span is 2x this value

@schedule(max_iters=5, wait_time=5, time_unit="days")
"""
schedule params:

max_iters: int, specifies how many times to run the collection
wait_time: int, specifies how many time_unit to wait before the next collection
time_unit: str, see scheduler.py for acceptable args
"""
def main():
        for topic in queries:
                path = f"/data/{topic}"

                if not os.path.exists(path):
                        os.mkdir(path)

                logging.basicConfig(filename='./logs/main.log', format="%(asctime)s - %(message)s",
                                level=logging.INFO)

                logging.info(f"{datetime.now().isoformat()} - Getting {topic.upper()}")

                cur_date = datetime.now().strftime("%b_%d").lower()

                print(f"Performing collections for {topic.upper()} on {cur_date}\n{15*'-'}")

                q = queries[topic]["q"]
                foc_date = datetime.fromisoformat(queries[topic]["focal_date"])
                start_date = foc_date - timedelta(days=onetailed_span)
                start_date = start_date.isoformat()[:19] + "Z"
                end_date = foc_date + timedelta(days=onetailed_span)
                end_date = end_date.isoformat()[:19] + "Z"

                collect_query = {
                "part": "snippet",
                "maxResults": 50,
                "order": "date",
                "safeSearch": "none",
                "publishedAfter": start_date,
                "publishedBefore": end_date,
                "type": "video",
                "q": q
                }

                video_file = f"{cur_date}_videos.ndjson"

                ytapi.collect_videos(query=collect_query, dev_key=dev_key, path=path, output_file=video_file, metadata_file=f"{cur_date}_metadata.ndjson", increment_calls=1, suppress_quota_warning=False, logfile=f"./logs/{cur_date}.log")

                vid_ids = set()
                channel_ids = set()
                with open(os.path.join(path, video_file), 'r') as f:
                        for line in f:
                                raw = json.loads(line)
                                vid_ids.add(raw['id']['videoId'])
                                channel_ids.add(raw['snippet']['channelId'])
                
                dets_query = {"part": "snippet,contentDetails,statistics", "id": vid_ids, "maxResults": 50}
                ytapi.get_video_details(query=dets_query, dev_key=dev_key, path=path, output_file=f"{cur_date}_details.ndjson", logfile=f"./logs/{cur_date}.log")

                chan_query = {"part": "snippet,contentDetails,statistics", "id": channel_ids, "maxResults": 50}
                ytapi.get_channel_details(query=chan_query, dev_key=dev_key, path=path, output_file=f"{cur_date}_channels.ndjson", logfile=f"./logs/{cur_date}.log")

                thread_query = {"part": "snippet,replies", "videoId": vid_ids, "maxResults": 100, "order": "time"}
                thread_file = f"{cur_date}_threads.ndjson"
                ytapi.collect_threads(query=thread_query, dev_key=dev_key, path=path, output_file=thread_file, logfile=f"./logs/{cur_date}.log")
                
                thread_ids = set()

                with open(os.path.join(path, thread_file), 'r') as f:
                        for line in f:
                                raw = json.loads(line)
                                if raw['snippet']['totalReplyCount'] > 5:
                                        thread_ids.add(raw['id'])
                
                comment_query = {"part": "id,snippet", "parentId": thread_ids, "maxResults": 100}
                ytapi.collect_comments(query=comment_query, dev_key=dev_key, output_file=f"{cur_date}_comments.ndjson", path=path, logfile=f"./logs/{cur_date}.log")


if __name__ == '__main__':
        main()
