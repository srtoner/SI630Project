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

    with open("placeids.txt", "r") as place_file:
        places_ids = place_file.readlines()

    placesids = [p.strip() for p in places_ids] 

    with open("places.pkl", "rb") as file:
        places = pkl.load(file)

    bad_places = []

    if os.path.exists("badplaceids.txt"):
        with open("badplaceids.txt", "r") as place_file:
            temp = place_file.readlines()
            bad_places = [p.strip() for p in temp]

    for pid in placesids:
        if not places.get(pid):
            if pid in bad_places:
                continue
            try:
                places[pid] = api.geo_id(pid)
            except:
                bad_places.append(pid)
                print("Rate Limit exceeded at {}".format(time.localtime()))
                print("Waiting 15 min")
                with open('places.pkl', 'wb') as f:
                    print("Length of places: {}".format(len(places)))
                    pkl.dump(places, f)

                with open("badplaceids.txt", "w") as place_file:
                    line_breaks = [t + '\n' for t in bad_places]
                    place_file.writelines(line_breaks)

                time.sleep(1000)

    with open('places.pkl', 'wb') as f:
        print("Length of places: {}".format(len(places)))
        pkl.dump(places, f)

    with open("badplaceids.txt", "w") as place_file:
        place_file.writelines(bad_places)

 