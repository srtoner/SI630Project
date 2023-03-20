import pandas as pd
import numpy as np
import requests
import os
import json
import pandas as pd
import pickle as pkl
#import TwitterUtilsEvergreen as tue

bearer_token = os.environ.get("BEARER_TOKEN")

def collect_user_ids(): 

    with open("users.json", "r") as file:
        user_json = file.read()

    test = '{"total": [' + user_json.replace("}{", "},{") + "]}"
    user_data = json.loads(test)
    users = [u['data'] for u in user_data["total"]]
    flat_list = [user_id for user in users for user_id in user]
    
    users_df = pd.DataFrame(flat_list)

    users_df = users_df.drop(['withheld'], axis = 1)

    return users_df

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserTweetsPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def collect_tweets(df, params): 
    
    overall_twitter_list = []

    user_ids = list(df['id'])

    for id in user_ids[:100]: 

        # user_dict = {}

        url = "https://api.twitter.com/2/users/{}/tweets".format(id)

        json_response = connect_to_endpoint(url, params)
        tweets = json_response['data']

        count = 0
        
        for tweet in tweets: 
            if not tweet.get("geo"):
                continue
            # print(tweet)
            tweet_dict = {}
            tweet_dict['user_id'] = id
            tweet_dict['tweet_id'] = tweet['id']
            tweet_dict['tweet_text'] = tweet['text']
            tweet_dict['place_id'] = tweet['geo']['place_id']

            overall_twitter_list.append(tweet_dict)
            # print("user_id: ", id, "tweet id: ", tweet['id'], "tweet text: ", tweet['text'], "place id: ", tweet['geo']['place_id'])
            # print(json.dumps(json_response, indent=4, sort_keys=True))
            count += 1
        
            if count > 50: 
                break

        # user_dict['user_id'] = tweet_list


        # overall_twitter_list.append(user_dict)

    print(overall_twitter_list)
    
    print("DONE")

    return overall_twitter_list


def main():

    df = collect_user_ids()
    params = {"tweet.fields": "id,text,geo"}
    tweets_list = collect_tweets(df, params)

    print(len(tweets_list))

    final = json.dumps(tweets_list, indent=2)

    with open("tweets-with-place.json", "w") as outfile:
        outfile.write(final)
    # print(df.head())
    # print("Shape of dataframe: ", df.shape)

if __name__ == "__main__":
    main()
