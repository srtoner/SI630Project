import pymongo
from pymongo import MongoClient
import json

import pickle as pkl


# Gets all users from seed sample and finds past tweet history

client = MongoClient('localhost', 27017)
db = client.test # Load test database 
collection = db.test # Within database, have collection

user_collection = db.users

sample_set = db.test.distinct('data.id')

samples = []

# Query through sample set
for s in sample_set:
    samples.append(s)
    

# Export 
with open("user_seed.pkl", "wb") as f:
    pkl.dump(samples, f)

# TODO: Implement proper filtering for diverse dataset.
# Could (and likely should) involve processing the script 
# over different time intervals to capture a full day for
# for each geo

# OR use a 

# Next Steps

# From Sampled ID set, collect XX tweets from each user


