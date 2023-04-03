import requests
import os
import json
import copy
import pandas as pd
import pickle as pkl

import time

import tweepy


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



# ENDPOINTS

BASE = 'https://api.twitter.com/2/'
GEO = 'https://api.twitter.com/1.1/'

default_rules = [{"value": "(a OR the) has:geo lang:en tweets_count:50", "tag": "active_user"}]

class TwitterClient:
    def __init__(self,**kwargs):
        # To set your enviornment variables in your terminal run the following line:
        # export 'BEARER_TOKEN'='<your_bearer_token>'
        self.bearer_token = os.environ.get("BEARER_TOKEN")
        self.stream = None
        with open('my_oauth.json', 'r') as f:
            self.oauth_tokens = json.load(f)
        self.sample = []
        rules = self.get_rules()
        delete = self.delete_all_rules(rules)
        set = self.set_rules(delete)
       

# Authentication
    def bearer_oauth(self, r):
        """
        Method required by bearer token authentication.
        """

        r.headers["Authorization"] = f"Bearer {self.bearer_token}"
        r.headers["User-Agent"] = "v2FilteredStreamPython"
        return r

# Managing Rules
    def get_rules(self):
        response = requests.get(
            "https://api.twitter.com/2/tweets/search/stream/rules", auth=self.bearer_oauth
        )
        if response.status_code != 200:
            raise Exception(
                "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
            )
        print(json.dumps(response.json()))
        return response.json()


    def delete_all_rules(self, rules):
        if rules is None or "data" not in rules:
            return None

        ids = list(map(lambda rule: rule["id"], rules["data"]))
        payload = {"delete": {"ids": ids}}
        response = requests.post(
            "https://api.twitter.com/2/tweets/search/stream/rules",
            auth=self.bearer_oauth,
            json=payload
        )
        if response.status_code != 200:
            raise Exception(
                "Cannot delete rules (HTTP {}): {}".format(
                    response.status_code, response.text
                )
            )
        print(json.dumps(response.json()))
        return response.json()

    def set_rules(self, delete):
        # You can adjust the rules if needed
        sample_rules = [
            {"value": "dog has:images", "tag": "dog pictures"},
            {"value": "cat has:images -grumpy", "tag": "cat pictures"},
        ]
        payload = {"add": default_rules}
        response = requests.post(
            "https://api.twitter.com/2/tweets/search/stream/rules",
            auth=self.bearer_oauth,
            json=payload,
        )
        if response.status_code != 201:
            raise Exception(
                "Cannot add rules (HTTP {}): {}".format(response.status_code, response.text)
            )
        print(json.dumps(response.json()))

    def connect_to_endpoint(self, url, params):
        response = requests.request("GET", url, auth=self.bearer_oauth, params=params)
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()
    
    def get_geo(self, url, params):
        response = requests.request("GET", url + params + '.json', auth=self.bearer_oauth)
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()

# Processing Stream
    def get_stream(self, set, endpoint = 'tweets/search/', sample_size = 10, keep_sample = False):
        if not keep_sample:
            self.sample = []

        response = requests.get(
            BASE + endpoint + 'stream', auth=self.bearer_oauth, stream=True , params = {'tweet.fields' : 'author_id,context_annotations,geo'}
        )
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(
                "Cannot get stream (HTTP {}): {}".format(
                    response.status_code, response.text
                )
            )
        
        for response_line in response.iter_lines():
            if response_line:
                json_response = json.loads(response_line)
                self.sample.append(json_response)
                sample_size -= 1
            if not sample_size:
                break

        return self.sample

    def collect_user_ids(self, file_path): 

        with open(file_path, "r") as file:
            user_json = file.read()

        wrapper = '{"total": [' + user_json.replace("}{", "},{") + "]}"
        user_data = json.loads(wrapper)
        users = [u['data'] for u in user_data["total"]]
        flat_list = [user_id for user in users for user_id in user]
        users_df = pd.DataFrame(flat_list)
        users_df = users_df.drop(['withheld'], axis = 1)

        return users_df

    def collect_tweets(self, df, params, processed_ids): 

        json_response = {'data' : []}
        overall_twitter_list = []
        user_ids = list(df['id'])
        for id in user_ids:
            if int(id) in processed_ids:
                continue

            url = "https://api.twitter.com/2/users/{}/tweets".format(id)
            try:
                json_response = self.connect_to_endpoint(url, params)
            except:
                print("Rate Limit exceeded at {}".format(time.localtime()))
                print("Waiting 15 min")
                with open("user_data_temp.pkl", "wb") as file:
                    pkl.dump(overall_twitter_list, file)
                time.sleep(900)

            tweets = json_response.get('data')

            if not tweets:
                continue

            count = 0
            
            for tweet in tweets: 
                if not tweet.get("geo"):
                    continue
                
                tweet_dict = {}
                tweet_dict['user_id'] = id
                tweet_dict['tweet_id'] = tweet['id']
                tweet_dict['tweet_text'] = tweet['text']
                tweet_dict['place_id'] = tweet['geo'].get('place_id')
                tweet_dict['convo_id'] = tweet['conversation_id']

                overall_twitter_list.append(tweet_dict)
                count += 1
            
                if count > 50: 
                    break
            
            processed_ids.update([int(id)])
            
        with open("processed_users.txt", "w") as file:
        # If process is interrupted, don't redundantly sample same users
            file.writelines(processed_ids)
        

        print(overall_twitter_list)
        
        print("DONE")

        return overall_twitter_list

    def collect_convos(self, df, params, processed_ids): 

        json_response = {'data' : []}
        overall_twitter_list = []
        tweet_ids = list(df['tweet_id'])
        for id in tweet_ids:
            if int(id) in processed_ids:
                continue
            
            kernel_url = BASE + 'tweets/'
            kernel_params = {'tweet.fields' : 'author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets'}
            kernel_params['ids'] = id
            try:
                kernel_tweet = self.connect_to_endpoint(kernel_url, kernel_params).get('data')
                
                params['query'] = 'conversation_id:' + kernel_tweet[0]['conversation_id']
                url = "https://api.twitter.com/2/tweets/search/all"

                json_response = self.connect_to_endpoint(url, params)
                time.sleep(1)
            except:
                print(json_response)
                print("Rate Limit exceeded at {}".format(time.localtime()))
                print("Waiting 15 min")
                with open("processed_tweets.txt", "w") as file:
                # If process is interrupted, don't redundantly sample same users
                    file.writelines([str(pid) for pid in processed_ids])

                with open("user_data_temp.pkl", "wb") as file:
                    pkl.dump(overall_twitter_list, file)
                time.sleep(900)

            tweets = json_response.get('data')

            if not tweets:
                continue

            count = 0
            
            for tweet in tweets: 
                tweet_dict = {}
                tweet_dict['user_id'] = tweet['author_id']
                tweet_dict['tweet_id'] = tweet['id']
                tweet_dict['tweet_text'] = tweet['text']
                tweet_dict['referenced_tweets'] = tweet['referenced_tweets']
                tweet_dict['convo_id'] = tweet['conversation_id']
                tweet_dict['reply_to_user_id'] = tweet['in_reply_to_user_id']

                overall_twitter_list.append(tweet_dict)
                count += 1
            
                if count > 50: 
                    break
            
            processed_ids.update([int(id)])
            
        with open("processed_tweets.txt", "w") as file:
        # If process is interrupted, don't redundantly sample same users
            file.writelines([str(pid) + '\n' for pid in processed_ids])

        print(overall_twitter_list)
        
        print("DONE")

        return overall_twitter_list
        
if __name__ == "__main__":
    print("Shalom")
    # test = TwitterClient()
    # rules = test.get_rules()
    # sample = test.get_stream(rules, sample_size=50)
    # user_params = {'user.fields' : 'location,name,description'}
    # #  user.derived.location.geo
    # tweet_locations = [s['data'].get('geo').get('place_id') for s in sample]
    
    # user_ids = [tweet['data']['author_id'] for tweet in sample]
    # user_params['ids'] = ','.join(user_ids)
    # user_endpoint = BASE + "users/"
    # user_test = test.connect_to_endpoint(user_endpoint, user_params)
    # print("Pause") 

    # # Concatentate users.json with existing dict

    # # with open('users.json', 'r') as f:
    # #     print("Length of users Regular Process): {}".format(len(user_test)))
    # #     json.dump(user_test, f)
    # rate_limit = False
    # location_params = {}
    # location_endpoint = GEO + "geo/id/"
    # location_params = '.json,'.join(tweet_locations)
    # places = {}

    # if os.path.exists("places.pkl"):
    #     with open('places.pkl', 'rb') as f:        
    #         places = pkl.load(f)

    # # Beware of exceeding the rate limit
    # # TODO: Create some sort of script to wait until timeout complete
    # for place in tweet_locations:
    #    if not places.get(place):
    #         time.sleep(3)
    #         try:
    #             places[place] = api.geo_id(place)
    #         except:
    #             # Timeout
    #             rate_limit = True
    #             print("Length of places: {}".format(len(places)))
    #             with open('places.pkl', 'wb') as f:
    #                 pkl.dump(places, f)
    #             break
        
    # print("Pause")
    # if not rate_limit:
    #     with open('places.pkl', 'wb') as f:
    #         print("Length of places (No Rate Limit): {}".format(len(places)))
    #         pkl.dump(places, f)
    
    # with open('users.json', 'a+') as f:
    #     print("Length of users (Regular Process): {}".format(len(user_test)))
    #     json.dump(user_test, f)
