import re
import time
import tweepy
from tweepy import OAuthHandler
import csv

class TwitterClient(object):
    
    def __init__(self):
        
        consumer_key = "W2M2ii8yFBEkpMcMToyhYT7pO"
        consumer_secret = "M7t8hBY7S5Gr7858rme3JYMV8OPwvdXB0jA0k1KStho6Wv08xp"
        access_token = "773234441485291520-91Dgd8CHkqvfjD4xpmtP2i0LnFl16dt"
        access_token_secret = "kTd052KWsOs2FCvKixEEnklMDLUqOY0Huv4NR0k1wpC5q"

        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")


    def clean_tweet(self, tweet):
        return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+://\S+)", " ", tweet).split())


    def get_full_tweet(self, id):        
        return self.api.get_status(id, tweet_mode="extended")


    def get_tweets(self, query, count = 10):
        tweets = []

        try:
            fetched_tweets = self.api.search_tweets(q = query, count = count, lang = "en")
            return fetched_tweets
            
        except tweepy.errors.TweepyException as e:
            print("Error : " + str(e))

def main():
    api = TwitterClient()

    f = open("extracted_tweets_racism7.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["Tweets", "Target"])

    tweets = api.get_tweets(query = "Racism", count = 500)
    
    written_tweets = []

    for tweet in tweets:
        tweet_id = tweet.id
        full_tweet = api.get_full_tweet(tweet_id)
        if "retweeted_status" in full_tweet._json.keys():
            if str(full_tweet._json["retweeted_status"]["full_text"]).strip() not in written_tweets:
                writer.writerow([str(full_tweet._json["retweeted_status"]["full_text"]).strip(), "Gun Laws"])
                written_tweets.append(str(full_tweet._json["retweeted_status"]["full_text"]).strip())

    f.close()


main()