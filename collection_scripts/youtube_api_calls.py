import json
import logging
import os
import warnings
import math
import googleapiclient.discovery
from googleapiclient.errors import HttpError
from tenacity import retry, wait_exponential, retry_if_exception_type, stop_after_delay, before_sleep_log
from tqdm import tqdm
from pytz import timezone as tz
from datetime import datetime, timedelta
from typing import Literal

logger = logging.getLogger(__name__)


def make_request(client, query, endpoint: Literal['search_list', 'video_list', 'threads', 'comments', 'channel']):
    if endpoint == 'search_list':
        # takes a parameter dict as argument
        req = client.search().list(**query)

    elif endpoint == 'video_list':
        # takes a list or set of video IDs as argument
        req = client.videos().list(**query)

    elif endpoint == 'threads':
        # takes a list or set of video IDs as argument
        req = client.commentThreads().list(**query)

    elif endpoint == 'comments':
        # takes a list of thread IDs as argument
        req = client.comments().list(**query)

    elif endpoint == 'channel':
        # takes a list of channel IDs as argument
        req = client.channels().list(**query)

    return req


@retry(retry=retry_if_exception_type(googleapiclient.errors.HttpError), wait=wait_exponential(multiplier=1, min=2, max=8), stop=stop_after_delay(20), before_sleep=before_sleep_log(logger, logging.WARNING))
def get_response(request):
    try:
        response = request.execute()
    except (AttributeError, HttpError) as e:
        if isinstance(e, HttpError):
            if e.status_code == 403 and e.error_details[0]['reason'] in ['quotaExceeded', 'rateLimitExceeded']:
                raise
            else:
                logging.info(f"Error: {e}")
                response = None
        else:
            logging.info(f"Error: {e}")
            response = None
        
    return response


def collect_videos(query, dev_key: str, output_file: str, metadata_file: str, logfile=None, increment_calls=None, path: str=None, suppress_quota_warning=True):

    if path:
        output_file = os.path.join(path, output_file)
        metadata_file = os.path.join(path, metadata_file)

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=dev_key)
    
    query = query.copy()

    try:
        end_date = query['publishedBefore']
    except KeyError as e:
        tz_pacific = tz('US/Pacific')
        end_date = datetime.now(tz=tz_pacific).isoformat()[:19] + "Z"  # YouTube platform (NOT the API) operates in PST
        query['publishedBefore'] = end_date
        warnings.warn(f"No 'publishedBefore' value provided in query. Defaulting to current time (PST): {end_date}", stacklevel=2)

    if 'publishedAfter' not in query:
        published_after = datetime.fromisoformat(query['publishedBefore'].replace("Z", "+00:00"))
        published_after -= timedelta(hours=168)
        query['publishedAfter'] = published_after.isoformat()[:19] + "Z"
        warnings.warn(f"No 'publishedAfter' value provided in query. Defaulting to one week prior to 'publishedBefore': {query['publishedAfter']}", stacklevel=2)    

    if logfile:
        logging.basicConfig(filename=logfile, format="%(asctime)s - %(message)s", level=logging.INFO)

    with open(output_file, 'w+') as fw, open(metadata_file, 'w+') as md:
        
        if increment_calls:
            end_date = query['publishedBefore']
            published_after = query['publishedAfter']
            published_before = datetime.fromisoformat(published_after.replace("Z", "+00:00"))
            published_before += timedelta(hours=increment_calls)
            published_before = published_before.isoformat()[:19] + "Z"
            query['publishedBefore'] = published_before
            
            timespan = datetime.fromisoformat(end_date) - datetime.fromisoformat(query['publishedAfter'])
            timespan = timespan.total_seconds()/3600  # convert to hours
            total_calls = math.ceil(timespan/increment_calls)  # round any decimals up to integers

            if not suppress_quota_warning:
                warnings.warn(f"This video collection operation will cost a minimum of {total_calls * 100} quota units. Please ensure you have enough to avoid rate limits, or limit your number of queries.")

            with tqdm(total=total_calls, desc="Collecting videos...") as pbar:
                while datetime.fromisoformat(published_after.replace("Z", "+00:00")) < datetime.fromisoformat(end_date.replace("Z", "+00:00")):
                    
                    request = make_request(youtube, query, endpoint='search_list')
                    # request = youtube.search().list(**query)

                    while True:
                        # response = request.execute()
                        response = get_response(request)
                        try:
                            items = response.pop('items')  # remove items from response dict so we can write it as metadata
                        except (KeyError, TypeError):
                            break

                        query_time = datetime.now(tz=tz('UTC'))
                        response['query_time'] = query_time.isoformat()[:19] + "Z"
                        response['query'] = query

                        md.write(json.dumps(response) + '\n')

                        # loop to write data
                        for item in items:
                            fw.write(json.dumps(item) + '\n')

                        if 'nextPageToken' in response:
                            query['pageToken'] = response['nextPageToken']
                            request = make_request(youtube, query, endpoint='search_list')
                            # request = youtube.search().list(**query)

                            # pg_count += 1
                            # if pg_count % 10 == 0:
                            #     print(f"Parsed {pg_count} pages")
                        else:
                            published_after = datetime.fromisoformat(published_after.replace("Z", "+00:00"))
                            published_after += timedelta(hours=increment_calls)
                            published_after = published_after.isoformat()[:19] + "Z"
                            query['publishedAfter'] = published_after
                            published_before = datetime.fromisoformat(published_before.replace("Z", "+00:00"))
                            published_before += timedelta(hours=increment_calls)
                            published_before = published_before.isoformat()[:19] + "Z"
                            query['publishedBefore'] = published_before

                            pbar.update(1)

                            break

        else:
            request = make_request(youtube, query, endpoint='search_list')
            # request = youtube.search().list(**query)

            while True:
                # response = request.execute()
                response = get_response(request)
                try:
                    items = response.pop('items')
                except (KeyError, TypeError):
                    break
                
                query_time = datetime.now(tz=tz('UTC'))
                response['query_time'] = query_time.isoformat()[:19] + "Z"
                response['query'] = query

                md.write(json.dumps(response) + '\n')

                # loop to write data
                for item in items:
                    fw.write(json.dumps(item) + '\n')

                if 'nextPageToken' in response:
                    query['pageToken'] = response['nextPageToken']
                    request = make_request(youtube, query, endpoint='search_list')
                    # request = youtube.search().list(**query)
                
                else:
                    break


def get_video_details(query, dev_key: str, output_file: str, logfile=None, path: str=None, ids=None):
    if path:
        output_file = os.path.join(path, output_file)

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=dev_key)

    if logfile:
        logging.basicConfig(filename=logfile, format="%(asctime)s - %(message)s", level=logging.INFO)

    if ids:
        video_ids = list(set(ids))
        query = query.copy()
        query['id'] = video_ids
    else:
        video_ids = list(set(query['id']))

    with open(output_file, 'w+') as fw:
        if len(query['id']) > query['maxResults']:
            try:
                window = query['maxResults']
            except KeyError:
                window = 5  # API default

            start_index = 0
            end_index = window

            print(f"Estimated quota cost: {math.ceil(len(video_ids)/window)}")

            with tqdm(total=len(video_ids), desc="Collecting video details") as pbar:
                while end_index < len(video_ids):
                    rolling_ids = video_ids[start_index:end_index]
                    temp_idq = ','.join(rolling_ids)

                    temp_query = query.copy()
                    temp_query['id'] = temp_idq

                    request = make_request(youtube, temp_query, endpoint='video_list')
                    response = get_response(request)
                    # response = request.execute()

                    start_index += window
                    end_index += window

                    pbar.update(window)

                    if response is None:
                        continue

                    for item in response['items']:
                        fw.write(json.dumps(item) + '\n')

                if pbar.n < len(video_ids):
                    pbar.update(len(video_ids)-pbar.n)

        else:
            temp_idq = ','.join(video_ids)
            temp_query = query
            temp_query['id'] = temp_idq

            request = make_request(youtube, temp_query, endpoint='video_list')
            response = get_response(request)
            # response = request.execute()

            if response is not None:
                for item in response['items']:
                    fw.write(json.dumps(item) + '\n')


def get_channel_details(query, dev_key: str, output_file: str, logfile=None, path: str=None, ids=None):
    if path:
        output_file = os.path.join(path, output_file)

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=dev_key)

    if logfile:
        logging.basicConfig(filename=logfile, format="%(asctime)s - %(message)s", level=logging.INFO)

    if ids:
        channel_ids = list(set(ids))
    else:
        channel_ids = list(set(query['id']))

    with open(output_file, 'w+') as fw:
        if len(query['id']) > query['maxResults']:
            try:
                window = query['maxResults']
            except KeyError:
                window = 5  # API default

            start_index = 0
            end_index = window

            print(f"Estimated quota cost: {math.ceil(len(channel_ids)/window)}")

            with tqdm(total=len(channel_ids), desc="Collecting channel details") as pbar:
                while end_index < len(channel_ids):
                    rolling_ids = channel_ids[start_index:end_index]
                    temp_idq = ','.join(rolling_ids)

                    temp_query = query.copy()
                    temp_query['id'] = temp_idq

                    request = make_request(youtube, temp_query, endpoint='channel')
                    response = get_response(request)
                    # response = request.execute()

                    start_index += window
                    end_index += window

                    pbar.update(window)

                    if response is None:
                        continue

                    for item in response['items']:
                        fw.write(json.dumps(item) + '\n')

                if pbar.n < len(channel_ids):
                    pbar.update(len(channel_ids)-pbar.n)

        else:
            temp_idq = ','.join(channel_ids)
            temp_query = query
            temp_query['id'] = temp_idq

            request = make_request(youtube, temp_query, endpoint='channel')
            response = get_response(request)
            # response = request.execute()

            if response is not None:
                for item in response['items']:
                    fw.write(json.dumps(item) + '\n')


def collect_threads(query, dev_key, output_file: str, path: str=None, logfile=None, ids=None):
    if path:
        output_file = os.path.join(path, output_file)

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=dev_key)

    if ids:
        video_ids = list(set(ids))
    else:
        video_ids = list(set(query['videoId']))

    if logfile:
        logging.basicConfig(filename=logfile, format="%(asctime)s - %(message)s", level=logging.INFO)
    
    query = query.copy()

    with open(output_file, 'w+') as fw:
        for idx in tqdm(video_ids):
            query['videoId'] = idx

            while True:
                request = make_request(youtube, query, endpoint='threads')
                response = get_response(request)
                # response = request.execute()
                if response is None:
                    break
                # request = make_request(youtube, query, endpoint='threads')
                # try:
                #     response = request.execute()
                # except AttributeError:
                #     print(f"{idx} not found, moving on...")
                #     break

                for item in response['items']:
                    fw.write(json.dumps(item) + '\n')
                    
                if 'nextPageToken' in response:
                    query['pageToken'] = response['nextPageToken']

                else:
                    try:
                        del query['pageToken']
                    except KeyError:
                        pass
                    break


def collect_comments(query, dev_key, output_file: str, path: str=None, logfile=None, ids=None):
    if path:
        output_file = os.path.join(path, output_file)

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=dev_key)

    if ids:
        thread_ids = list(set(ids))
    else:
        thread_ids = list(set(query['parentId']))

    if logfile:
        logging.basicConfig(filename=logfile, format="%(asctime)s - %(message)s", level=logging.INFO)
    
    query = query.copy()

    with open(output_file, 'w+') as fw:
        for idx in tqdm(thread_ids):
            query['parentId'] = idx
            pgcount = 0

            while True:
                request = make_request(youtube, query, endpoint='comments')
                response = get_response(request)
                # response = request.execute()
                if response is None:
                    break
                # request = make_request(youtube, query, endpoint='threads')
                # try:
                #     response = request.execute()
                # except AttributeError:
                #     print(f"{idx} not found, moving on...")
                #     break

                for item in response['items']:
                    fw.write(json.dumps(item) + '\n')
                    
                if 'nextPageToken' in response:
                    query['pageToken'] = response['nextPageToken']

                else:
                    try:
                        del query['pageToken']
                    except KeyError:
                        pass
                    break
