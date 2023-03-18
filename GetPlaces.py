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

    with open("placeids.txt", "a+") as place_file:
        placesids = place_file.readlines()

    with open("places.pkl", "rb") as file:
        places = pkl.load(file)

    for pid in placesids:
        if not places.get(pid):
            try:
                places[pid] = api.geo_id(pid)
            except:
                print("Rate Limit exceeded at {}".format(time.localtime()))
                print("Waiting 15 min")
                time.sleep(900)

    with open('places.pkl', 'wb') as f:
        print("Length of places: {}".format(len(places)))
        pkl.dump(places, f)

