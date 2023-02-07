import pymongo
from pymongo import MongoClient
import json
client = MongoClient()

client = MongoClient('localhost', 27017)
db = client.test # Load test database (may need to be done externally, can write bash script) - Steve


file_range = [i for i in range(10)]
json_files = [str(f) + '.json' for f in file_range]

json_input = []

# Untested - Steve
for file in json_files:
    with open(file) as f:
        temp = json.load(f)
        json_input.append(temp)

db.insert_many(json_input)

