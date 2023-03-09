from pandas import DataFrame
import twint
import csv
import nest_asyncio
import time
nest_asyncio.apply()

topics = {"covid": ["covid", "covid19", "covid-19", "covid 19", "coronavirus", "omicron", "delta", "masks", "vaccines", "booster shot"], "climate": ["global warming", "climate change"], "immigration": ["immigration"], "healthcare": ["healthcare"], "guns": ["gun control", "gun violence"], "Biden": ["Biden"], "Trump": ["Trump"] }

n = 0
with open('labeled_users.csv', newline='') as csvfile, open('missingUsers', 'w') as missingUsersFile:
    reader = csv.reader(csvfile)
    for row in reader:

        #create new dataframe for each congressperson
        scraped_tweets = DataFrame()
        label, username = row
        userMissing = False

        for topic, keywords in topics.items():
            time.sleep(1)
            search_string = " OR ".join(keywords)

            # configure searchs
            c = twint.Config()
            c.Search = topic
            c.Pandas = True
            c.Lang = 'en'
            c.Limit = 10000
            c.Username=username
            try:
                twint.run.Search(c)
            except ValueError as e:
                # stop scraping tweets for this user once we know they have no tweets
                missingUsersFile.write(username)
                userMissing = True
                break

            # append to dataframe
            test_df = twint.storage.panda.Tweets_df
            n += len(test_df)
            if len(test_df) != 0:
                tweet_info = test_df[["username", "tweet"]]
                tweet_info["topic"] = topic
                scraped_tweets = scraped_tweets.append(tweet_info)

        # once we have tweet data, write to file
        if not userMissing:     
            scraped_tweets.to_csv(f"scraped_tweets/{username}.csv", index=False)

# save results
print(n)

