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
    # The below three lines are now taken care of in the init()
    # call for client
    rules = client.get_rules()
    delete = client.delete_all_rules(rules)
    set = client.set_rules(delete)
    user_params = {'user.fields' : 'location,name,description'}

    for i in range(1000):
        # Sample 100 tweets at random
        try:
            sample = client.get_stream(set, sample_size = 100)
            user_ids = [tweet['data']['author_id'] for tweet in sample]
            # Each Tweet should have geo by default rules; 
            # sometimes it appears that this is blank however
            tweet_locations = [s['data'].get('geo').get('place_id') for s in sample]
            user_ids = [tweet['data']['author_id'] for tweet in sample]
            # We are exporting the locations for future use in GetPlaces.py
            with open("placeids.txt", "a+") as place_file:
                place_file.writelines(tweet_locations)

            user_params['ids'] = ','.join(user_ids)
            user_endpoint = client.BASE + "users/"
            users = client.connect_to_endpoint(user_endpoint, user_params)
            

        except:
            # Rate limit, wait for it to expire
            time.sleep(900)

        