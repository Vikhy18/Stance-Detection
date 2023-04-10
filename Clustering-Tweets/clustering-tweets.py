from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import csv


def cluster_tweets(text):
    vectorizer = TfidfVectorizer(stop_words={'english'})
    X = vectorizer.fit_transform(text)

    model = KMeans(n_clusters=3, init='k-means++', max_iter=500, n_init=20)
    model.fit(X)

    labels=model.labels_
    clusters=pd.DataFrame(list(zip(text,labels)),columns=['title','cluster'])
        
    return labels, clusters



if __name__ == "__main__":
    text = pd.read_csv("extracted_tweets_immigration.csv", encoding= 'unicode_escape')
    tweets = text["Tweets"]
    target = text["Target"]
    labels, clusters = cluster_tweets(tweets.to_list())

    tweets_upd = clusters["title"]
    cluster_no = clusters["cluster"]

    df = pd.concat([tweets_upd, target, cluster_no], axis=1)

    df.to_csv("labelled_immigration.csv", index=False)