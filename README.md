# ICS438--Final

Info:
This project contains a search function that returns the top 10 related articles using data from 2004. Some relevant searches may not appear due to the age. 


ICS 438 Final Project
Refs:
https://www.youtube.com/watch?v=6QCHz2fD8d8 
https://www.youtube.com/watch?v=D2V1okCEsiE
https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/
https://www.kaggle.com/code/funxexcel/c9-cosine-similarity 
https://thepythoncode.com/article/calculate-rouge-score-in-python
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/

Thought Process:
- Goal is to show the top 10 relevant or as related as possible to the search 
How?
- Convert the search terms into numbers > the only way I could match with the data
- Convert the data into numbers, to create the cluster, the algorithm must cluster the numbers by relevance/closeness to 
one another
Process 
- First process the data by creating the cluster using TF-IDF (convert them into numbers) 
- determine closeness by comparing the numerical value > this will determine the cluster
- determine the best K-value to use ( testing different values and testing which is the fastest)
- user search > turns into numbers using TF-IDF
- trained model will find the cluster and ignore anything else
- compares the query to the values in the cluster to give the return value
- returns the values