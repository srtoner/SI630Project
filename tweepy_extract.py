import tweepy
import os
import json
import pandas as pd

input_files = ['elites-data.csv',
			    'senate.csv',
                'elites-twitter-data.csv',
                'state-level-variables.csv',
                'house.csv',
                'voter-matches.csv']

tweet_collections = ['2012Election.txt'] # Just as proof of concept
                    #  'GunControl.txt',
                    #  'SOTU.txt',
                    #  'Boston.txt',
                    #  'MinimumWage.txt',
                    #  'SuperBowl.txt',
                    #  'Budget.txt',
                    #  'Olympics.txt',
                    #  'SupremeCourt.txt',
                    #  'GovtShutdown.txt',
                    #  'Oscars.txt',
                    #  'Syria.txt']

with open('my_oauth.json', 'r') as f:
    oauth_tokens = json.load(f)

input_data_dict = {input_file.replace('.csv', ''
                  ):pd.read_csv('input/'+input_file) 
                  for input_file in input_files} # Unnecessary - Steve

twt_collections = {}

topics = [t.replace('.txt', '') for t in tweet_collections]

twt_prefix = 'tweet-collections/'
for topic in tweet_collections:
    with open(twt_prefix + topic, 'r') as f:
        twt_collections[topic.replace('.txt', '')] = [line.strip() for line in f.readlines()]

ids = twt_collections['2012Election'][0:9] # Demo mode

# Authentication should be wrapped in a function
consumer_key = os.environ.get("CONSUMER_KEY")
consumer_secret = os.environ.get("CONSUMER_SECRET")
access_token = oauth_tokens['oauth_token']
access_token_secret = oauth_tokens['oauth_token_secret']

# tweepy workflow, not sure how it differs from twint
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.lookup_statuses(ids)
for idx, tweet in enumerate(public_tweets):
    with open(str(idx) + '.json', 'w') as f:
        json.dump(tweet._json, f)