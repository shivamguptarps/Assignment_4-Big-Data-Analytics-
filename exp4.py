import tweepy
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')

BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAABPVzgEAAAAAuNSrKvyoMj%2FparBx0HGSUBR%2BW4g%3DrSqermoeW4Lp4FRT2k1cf0Y79m0giOYYukLxjsdR9yBD57SCTg"

client = tweepy.Client(bearer_token=BEARER_TOKEN)

keywords = ["AI technology", "machine learning", "climate change", "crypto market", "sports news"]
num_tweets = 50


def fetch_tweets(query, count):
    tweets = client.search_recent_tweets(query=query, tweet_fields=["text"], max_results=count)
    return [tweet.text for tweet in tweets.data] if tweets.data else []

tweet_data = [tweet for keyword in keywords for tweet in fetch_tweets(keyword, num_tweets)]
df = pd.DataFrame(tweet_data, columns=["Tweet"])

glove_vectors = KeyedVectors.load_word2vec_format("glove.6B.300d.txt", binary=False, no_header=True)


def get_glove_vector(sentence):
    words = word_tokenize(sentence.lower())
    vectors = [glove_vectors[word] for word in words if word in glove_vectors]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

df["Vector"] = df["Tweet"].apply(get_glove_vector)
tweet_vectors = np.vstack(df["Vector"].values)

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["Cluster"] = kmeans.fit_predict(tweet_vectors)

cos_sim_matrix = cosine_similarity(tweet_vectors)
df["Cosine Similarity"] = [cos_sim_matrix[i].mean() for i in range(len(df))]

df.to_csv("clustered_tweets.csv", index=False)
print("\n Scraping, Embedding & Clustering Done! Results saved in clustered_tweets.csv")
