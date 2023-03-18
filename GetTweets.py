import requests
import os
import json
import pandas as pd
import pickle as pkl
import TwitterUtils as TU

# Copied over functions from collect_tweets.py

def collect_user_ids(): 
    with open("users.json", "r") as file:
        temp = file.read()
    user_json = '{"total": [' + temp.replace("}{", "},{") + "]}"
    user_data = json.loads(user_json)
    users = [u['data'] for u in user_data["total"]]
    
    flat_list = [user_id for user in users for user_id in user]
    
    users_df = pd.DataFrame(flat_list)
    users_df = users_df.drop(['withheld'], axis = 1)
    
    return users_df


if __name__ == "__main__":
    pass