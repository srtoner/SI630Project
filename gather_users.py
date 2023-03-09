import requests
import os
import json
import pickle as pkl

# See other demo files for how to get this VVV set up. It took me a minute, so lmk if you have any issues
# ~ Steve
bearer_token = os.environ.get("BEARER_TOKEN")


# /* Starter Code
def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FilteredStreamPython"
    return r

def get_rules():
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", auth=bearer_oauth
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))
    return response.json()

# NOTE: We should save all our rules to an external source. Twitter has been deleting
# rules for accounts not used very frequently

def delete_all_rules(rules):
    if rules is None or "data" not in rules:
        return None

    ids = list(map(lambda rule: rule["id"], rules["data"]))
    payload = {"delete": {"ids": ids}}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=bearer_oauth,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot delete rules (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    print(json.dumps(response.json()))
    ### Where is the return? Returns None in both cases, so wtf mates?

def set_rules(delete):
    # WTF? Why would we take an argument and never use it?
    sample_rules = [
        # Below is just an example of using a stop word 
        {"value": "(a OR the) has:geo lang:en tweets_count:50", "tag": "active_user"},
    ]
    payload = {"add": sample_rules}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=bearer_oauth,
        json=payload,
    )
    if response.status_code != 201:
        raise Exception(
            "Cannot add rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))


def get_stream(set):
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream", auth=bearer_oauth, stream=True,
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
            print(json.dumps(json_response, indent=4, sort_keys=True))


def test_user_stream(set):
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream", auth=bearer_oauth, stream=True,
    )
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    
    count = 0
    payload = []
    for response_line in response.iter_lines():
        if response_line:
            json_response = json.loads(response_line)
            payload.append(json_response)
            count += 1
        if count > 10:
            break

    
    print("pause")
    with open("user_exp.pkl", "wb") as file:
        pkl.dump(payload, file)

    

def main():
    rules = get_rules()
    delete = delete_all_rules(rules)
    set = set_rules(delete)
    test_user_stream(set)

# */ End Starter Code


# Create Rules for our purposes. We generally want (for first pass):
# Geo enabled
# More than a handful of tweets (reasonable threshold?)
# Verified (although bots and such could be interesting)
# English Speaking

# TODO: Create Utils.py that addresses all of the above authentication nonsense


def create_rule(values, tags, lang = ''):
    pass

if __name__ == "__main__":


    rules = [
        # Below is just an example of using a stop word 
        {"value": "(a OR the) has:geo lang:en tweets_count:50", "tag": "active_user"},
    ]
    main()