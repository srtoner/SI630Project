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

    # Load info on users themselves
    user_df = client.collect_user_ids("users.json")
    params = {"max_results": 100,"tweet.fields": "id,text,geo"}
    # Init empty on first run
    data = []
 
    processed_ids = set()

    with open("processed_users.txt", "r") as file:
        # If process is interrupted, don't redundantly sample same users
        processed_ids = set(file.readlines())     

    # data = client.collect_tweets(user_df, params, processed_ids)

    print("Pause")

        
    # with open("user_data_.pkl", "wb") as file:
    #     pkl.dump(data, file)
