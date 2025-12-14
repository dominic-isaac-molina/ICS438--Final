# imports
import time
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
from search import SearchEngine

# sample inputs for testing
test_queries = [
    "latest technology trends in smartphones",  
    "price of gold",                            
    "oil prices and stock market crash",        
    "new medical treatments for cancer",        
    "Formula 1 champion"                        
]

# function where calculation of test happens will return scores (Rouge)
def get_rouge_f1(candidate, reference):

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    
    return scores['rougeL'].fmeasure

# function where calculation of test happens will return scores (Bert)
def get_bert_f1(candidate, reference):

    P, R, F1 = score(
        [candidate], 
        [reference], 
        lang="en", 
        model_type="distilbert-base-uncased", 
        verbose=False
    )
    
    return F1.item()

# runs the test using the prompts above will return the avg scores
def run_evaluation():
    
    # initialize the engine 
    try:
        engine = SearchEngine()
    except Exception as e:
        print(f"Error initializing search engine: {e}")
        return

    # Data collection for averages
    metrics_log = {
        "time": [],
        "rouge": [],
        "bert": []
    }
    
    for query in test_queries:
        # search
        results, time_taken, cluster_id = engine.search(query, top_k=1)
        
        if not results:
            rouge_val, bert_val = 0.0, 0.0
            top_title = "No Result"
        else:
            top_result_text = results[0]['title']
            
            # calculate metrics
            rouge_val = get_rouge_f1(top_result_text, query)
            bert_val = get_bert_f1(top_result_text, query)

        # log Data
        metrics_log["time"].append(time_taken)
        metrics_log["rouge"].append(rouge_val)
        metrics_log["bert"].append(bert_val)
        
    # calculate averages
    avg_time = np.mean(metrics_log["time"])
    avg_rouge = np.mean(metrics_log["rouge"])
    avg_bert = np.mean(metrics_log["bert"])

    print("\nevaluation summary")
    print(f"average Search Time: {avg_time:.4f} seconds")
    print(f"average ROUGE-L F1:  {avg_rouge:.4f}")
    print(f"average BERTScore F1:{avg_bert:.4f}")

if __name__ == "__main__":
    run_evaluation()