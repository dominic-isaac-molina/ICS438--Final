# imports
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Path connection
output_folder = os.path.join(os.path.dirname(__file__), '..', 'output')
input_file = os.path.join(output_folder, 'metadata.parquet')

# Model paths (outputs)
vectorizer_file = os.path.join(output_folder, 'vectorizer.joblib')
kmeans_file = os.path.join(output_folder, 'kmeans.joblib')
vector_file = os.path.join(output_folder, 'vectors.joblib')

# parameters (kmeans value and the randomizer value)
num_clusters = 100
random_seed = 42

# create the cluster
def build_index():
    
    # Load the preprocessed data
    if not os.path.exists(input_file):
        # send error if not found (logs for debugging)
        print(f"Error {input_file} file not found")
        return

    # if found 
    df = pd.read_parquet(input_file)
    texts = df['cleaned_text'].tolist()
    
    # Transform words into number using TF-IDF
    # limiting max_features to 1000 keeps it fast titles aren't that long
    # stop_words will filter out common words like of, the, is > not useful when comparing 
    # want keywords not all words
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(texts)

    # group the clusters based on similarity
    # n_init set pretty high so we have the best fit possible
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed, n_init=25)
    kmeans.fit(X)
    
    # save cluster IDs back to the dataframe
    df['cluster_label'] = kmeans.labels_
    df.to_parquet(input_file) 

    # save the trained model
    # used for searching phase 
    joblib.dump(vectorizer, vectorizer_file)
    joblib.dump(kmeans, kmeans_file)
    joblib.dump(X, vector_file)

if __name__ == "__main__": build_index()