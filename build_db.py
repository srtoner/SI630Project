import pymongo
from pymongo import MongoClient
import json


# Mark for deletion?
import pickle as pkl
client = MongoClient()

client = MongoClient('localhost', 27017)
db = client.test # Load test database 
collection = db.test # Within database, have collection

# This needs to be populated with inputs from scraping with API
file_range = [i for i in range(3)]
json_files = [str(f) + '.json' for f in file_range]

json_input = []

# Untested - Steve
# for file in json_files:
#     with open(file) as f:
#         temp = json.load(f)
#         json_input.append(temp)

with open('user_test.pkl', 'rb') as f:
    json_input = pkl.load(f)
 

collection.insert_many(json_input)

