import tweepy
import os
import json

with open('my_oauth.json', 'r') as f:
    oauth_tokens = json.load(f)


consumer_key = os.environ.get("CONSUMER_KEY")
consumer_secret = os.environ.get("CONSUMER_SECRET")
access_token = oauth_tokens['oauth_token']
access_token_secret = oauth_tokens['oauth_token_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)