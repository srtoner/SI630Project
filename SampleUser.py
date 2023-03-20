import requests
import os
import json
import pandas as pd
import pickle as pkl
import TwitterUtils as TU
import time

if __name__ == "__main__":
    # Set Seed for sampling users
    client = TU.TwitterClient()

    user_params = {'user.fields' : 'location,name,description'}

    for i in range(1000):
        # Sample 100 tweets at random
        try:
            sample = client.get_stream(None, sample_size = 50)
            user_ids = [tweet['data']['author_id'] for tweet in sample]
            # Each Tweet should have geo by default rules; 
            # sometimes it appears that this is blank however
            tweet_locations = [s['data'].get('geo').get('place_id') for s in sample]
            user_ids = [tweet['data']['author_id'] for tweet in sample]
            # We are exporting the locations for future use in GetPlaces.py
            with open("placeids.txt", "a") as place_file:
                line_breaks = [t + '\n' for t in tweet_locations]
                place_file.writelines(line_breaks)

            user_params['ids'] = ','.join(user_ids)
            user_endpoint = TU.BASE + "users/"
            users = client.connect_to_endpoint(user_endpoint, user_params)
            with open('users.json', 'a+') as f:
                json.dump(users, f)
        except:
            # Rate limit, wait for it to expire
            print("Rate Limit exceeded at {}".format(time.localtime()))
            print("Waiting a min")
            time.sleep(1000)

    print("DONE")