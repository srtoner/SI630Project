# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3.8.8 ('base')
#     language: python
#     name: python3
# ---

# +
import requests
import os
import json
import pandas as pd
import numpy as np
import pickle as pkl
import TwitterUtils as TU
import seaborn as sns
import re

import spacy
import spacy_langdetect as sld

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="dark")
# -

df = pd.read_csv('filtered_data.csv')

# Slice by all available dimensions

df.columns

df['tweet_length'] = df['tweet_text'].apply(len)


# By Country

# +
def count_unique(x):
    return len(pd.unique(x))

def avg_length(x):
    return np.mean(len(x))


# -

# Unique Users by country

by_country = df.groupby('country').agg({'user_id':count_unique, 'tweet_id':count_unique, 'tweet_length':'mean'})
# by_country.sort_values(by = 'user_id', ascending=False).head(20)
by_country['Avg Tweets'] = by_country['tweet_id'] / by_country['user_id']
by_country

by_place = df.groupby('name').agg({'tweet_id':count_unique})
# by_country.sort_values(by = 'user_id', ascending=False).head(20)
by_place.sort_values(by='tweet_id', ascending=False).head(20)

# +

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="dark")
import pickle as pkl
from sklearn.decomposition import PCA
import nltk
# -

tokenizer = nltk.RegexpTokenizer(r'\w+')
df['tokenized_text'] = df['tweet_text'].apply(tokenizer.tokenize)


df.tokenized_text

import gensim.downloader
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')

# +
user_ids = pd.unique(df.user_id)[:1000]

by_user_dict = {}

for user in user_ids:

    user_df = df[df.user_id == user]

    user_tweets = [tt for tt in user_df.tokenized_text]
    user_embeddings = []
    for tweet in user_tweets:
        user_embeddings.append([])
        for token in tweet:
            try:
                user_embeddings[-1].append(glove_vectors[token])
            except:
                continue
        user_embeddings[-1] = np.array([embed for embed in user_embeddings[-1] if embed.shape]).mean(axis = 0)

    by_user_dict[user] = {
                        'country':user_df.country.iloc[0],
                        'tweets' : user_tweets,
                        'embeddings':user_embeddings,
                        'mean_embedding': np.array([embed for embed in user_embeddings if embed.shape]).mean(axis = 0)
                    }


# -

by_user_dict[2586324829]['mean_embedding'].shape

by_user_dict.keys()

mean_embeddings = np.array(by_user_dict[user]['embeddings'])

test = np.hstack(mean_embeddings)
test.shape

two_dim = mean_embeddings
xPCA = two_dim[:,0]; yPCA = two_dim[:,1]
xPCA.shape

yPCA

f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=xPCA, y=yPCA, s =5, color=".15")
sns.histplot(x=xPCA, y=yPCA, bins=50, pthresh=.1)
sns.kdeplot(x=xPCA, y=yPCA, levels=5, color="w", linewidths=1)

# The top 20 countries account for 96% of the total users collected, which suggests pretty good coverage. Dropping unnecessary columns

target_fields = ['user_id', 'tweet_id', 'tweet_text', 'place_id', 'name',
       'full_name', 'country', 'country_code', 'type', 'username',
       'description', 'user_name_field', 'location', 'withheld']

reduced_df = full_data[target_fields]

# Drop Duplicates

top20_countries = top20.index
top20 = reduced_df[reduced_df['country'].isin(top20_countries)]
unique_tweets = top20['tweet_id'].unique()
top20 = top20.drop_duplicates('tweet_id')



top20.to_csv('filtered_data.csv', index = False)


# +
def get_lang_detector(nlp, name):
    return sld.LanguageDetector()

# Uncomment when running for first time
nlp = spacy.load("en_core_web_sm")
spacy.Language.factory('language_detector', func = get_lang_detector)
nlp.add_pipe('language_detector', last =True)

def get_language(text):
    return nlp(text)._.language['score']


# -

data_dict = top20.to_dict(orient='records')

langs = [get_language(d['tweet_text']) for d in data_dict]

top_20_np = top20.to_numpy()

top20['tweet_text'].to_numpy()

top20['tweet_text'].to_numpy()

np.apply_over_axes(get_language, top20['tweet_text'].to_numpy(), [1])

top20.head()

