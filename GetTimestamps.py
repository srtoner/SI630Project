import requests
import os
import json
import pandas as pd
import numpy as np
import pickle as pkl
import TwitterUtils as TU
import seaborn as sns
import re
import tweepy
import time

with open('my_oauth.json', 'r') as f:
    oauth_tokens = json.load(f)


consumer_key = os.environ.get("CONSUMER_KEY")
consumer_secret = os.environ.get("CONSUMER_SECRET")
access_token = oauth_tokens['oauth_token']
access_token_secret = oauth_tokens['oauth_token_secret']

# tweepy workflow, not sure how it differs from twint
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

if __name__ == "__main__":


    client = TU.TwitterClient()

    with open("user_data_convo.pkl", "rb") as file:
        convos = pkl.load(file)
    convo_df = pd.DataFrame(convos)

    data_df = pd.read_csv('filtered_data.csv')

    original_ids = list(data_df.tweet_id.astype(str).dropna().unique())
    convo_ids = list(convo_df.tweet_id.astype(str).dropna().unique())
    replied_to_list = [c['referenced_tweets'][-1].get('id') for c in convos]
    convo_df['replied_to_id'] = replied_to_list
    replied_ids = list(convo_df.replied_to_id.astype(str).dropna().unique())


    tweet_id_set = set(replied_ids)
    tweet_id_set.update(original_ids)
    tweet_id_set.update(convo_ids)
    tweet_id_list = list(tweet_id_set)

    clean_ids = []

    for idx, elt in enumerate(tweet_id_list):
        try:
            clean_ids.append(str(np.int64(elt)))
        except:
            pass


    kernel_url = TU.BASE + 'tweets/'
    kernel_params = {'tweet.fields' : 'author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets'}
    

    timestamps = []


    for id_set in chunker(clean_ids, 99):

        kernel_params['ids'] = ",".join(id_set)
    
        try:
            output = client.connect_to_endpoint(kernel_url, kernel_params)
            timestamps.append(output)
        except:
            print("stop")
            time.sleep(900)
            try:
                output = client.connect_to_endpoint(kernel_url, kernel_params)
                timestamps.append(output)
            except:
                continue



    with open("timestamps.pkl", "wb") as file:
        pkl.dump(timestamps, file)

        