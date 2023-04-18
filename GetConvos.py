import requests
import os
import json
import pandas as pd
import pickle as pkl
import TwitterUtils as TU

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

if __name__ == "__main__":
    client = TU.TwitterClient()

    user_data = pd.read_csv('filtered_data.csv')

    # with open("user_data.pkl", "rb") as file:
    #     user_data = pd.DataFrame(pkl.load(file))
    params = {"tweet.fields": "author_id,text,geo,conversation_id,created_at,in_reply_to_user_id,referenced_tweets"}
    # Init empty on first run
    data = []
 
    processed_ids = set(user_data.tweet_id)

    with open("processed_tweets.txt", "r") as file:
        # If process is interrupted, don't redundantly sample same users
        processed_ids = set(file.readlines())     

    data = client.collect_convos(user_data, params, processed_ids)

    print("Pause")

        
    with open("user_data_convo.pkl", "wb") as file:
        pkl.dump(data, file)


        