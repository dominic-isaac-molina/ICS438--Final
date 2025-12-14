# imports
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import time

# Path connection
output_folder = os.path.join(os.path.dirname(__file__), '..', 'output')
metadata_file = os.path.join(output_folder, 'metadata.parquet')
vectorizer_file = os.path.join(output_folder, 'vectorizer.joblib')
KMEANS_PATH = os.path.join(output_folder, 'kmeans.joblib')
VECTORS_PATH = os.path.join(output_folder, 'vectors.joblib')

class SearchEngine:
    # function for loading data from training file
    def __init__(self):
        # Load all the training ouput
        self.df = pd.read_parquet(metadata_file)
        self.vectorizer = joblib.load(vectorizer_file)
        self.kmeans = joblib.load(KMEANS_PATH)
        self.doc_vectors = joblib.load(VECTORS_PATH)

    # function that perfroms the following tasks
    # find the cluster of the query 
    # search in that cluster
    def search(self, query, top_k=10):
        # for benchmark (time)
        start_time = time.time()
        
        # convert search string into numbers
        # locate which cluster to search in
        # locates the documents in the cluster and returns the position
        query_vec = self.vectorizer.transform([query])
        predicted_cluster = self.kmeans.predict(query_vec)[0]
        cluster_indices = self.df.index[self.df['cluster_label'] == predicted_cluster].tolist()
        # ignore everything else and focus on the cluster to speed up search
        cluster_vectors = self.doc_vectors[cluster_indices]

        # cosine similarity between the query and the cluster's articles
        # get indices of the top_k highest scores
        # filter only the high scores as those are the best value for top_k
        similarity_scores = cosine_similarity(query_vec, cluster_vectors).flatten()
        top_indices_local = similarity_scores.argsort()[-top_k:][::-1]
        
        # return value which are the title and scores for evaluation
        results = []
        for local_idx in top_indices_local:
            global_idx = cluster_indices[local_idx]
            score = similarity_scores[local_idx]
            
            record = self.df.iloc[global_idx]
            results.append({
                'title': record['title'],
                'description': record['description'],
                'score': float(score),
                'cluster': int(predicted_cluster)
            })
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        return results, execution_time, predicted_cluster
