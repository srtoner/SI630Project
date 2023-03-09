import requests
import os
import json
import copy

# Reference:
# Tweet Attributes: 



# ENDPOINTS

BASE = 'https://api.twitter.com/2/'

class TwitterClient:
    def __init__(self,**kwargs):
        # To set your enviornment variables in your terminal run the following line:
        # export 'BEARER_TOKEN'='<your_bearer_token>'
        self.bearer_token = os.environ.get("BEARER_TOKEN")
        self.stream = None
        with open('my_oauth.json', 'r') as f:
            self.oauth_tokens = json.load(f)
        self.sample = []
       

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


    def set_rules(self, delete):
        # You can adjust the rules if needed
        sample_rules = [
            {"value": "dog has:images", "tag": "dog pictures"},
            {"value": "cat has:images -grumpy", "tag": "cat pictures"},
        ]
        payload = {"add": sample_rules}
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

# Processing Stream
    def get_stream(self, set, endpoint = 'tweets/search/', sample_size = 10):
        response = requests.get(
            BASE + endpoint + 'stream', auth=self.bearer_oauth, stream=True , params = {'tweet.fields' : 'author_id,context_annotations'}
        )
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(
                "Cannot get stream (HTTP {}): {}".format(
                    response.status_code, response.text
                )
            )

            # self.stream = response
        for response_line in response.iter_lines():
            if response_line:
                json_response = json.loads(response_line)
                self.sample.append(json_response)
                sample_size -= 1
            if not sample_size:
                break

        return
    
if __name__ == "__main__":
    test = TwitterClient()
    rules = test.get_rules()
    sample = test.get_stream(rules)
    user_params = {'user.fields' : 'location,name,description'}
    print("Pause")
    user_ids = [tweet['data']['author_id'] for tweet in sample]
    user_params['ids'] = ','.join(user_ids)
    user_endpoint = BASE + "users/"
    user_test = test.connect_to_endpoint(user_endpoint, user_params)
    print("Pause")
