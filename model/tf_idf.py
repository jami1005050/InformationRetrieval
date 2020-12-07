'''
read document
Create a Term-Document Matrix with TF-IDF weighting
Write your queries and convert it as vector (based on TF-IDF)
Calculate the cosine similarity between the query and the document and repeat the process on each document.
Finally, show the document
By this method the performance evaluation is difficult.
'''

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from data_loader.data import Data
import numpy as np
data = Data()
data.readData()
document_array = []
for key in data.doc_set:
    print(data.doc_set[key])
    print("\n")
    document_array.append(data.doc_set[key])
query_array = []
for key in data.qry_set:
    query_array.append(data.qry_set[key])
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(document_array)

# Create a DataFrame
df = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names())
# print(df.head())
# print(df.shape)


def get_similar_articles(q, df):
    print("query:", q)
    print("The following are the articles with the highest cosine value: ")
    q = [q]
    q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0], )
    sim = {}
    for i in range(10):
        sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)

    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)

    for k, v in sim_sorted:
        if v != 0.0:
            print("Similarity:", v)
            # print(document_array[k])
            print(k)

    for key in data.rel_set:
        # print(key)
        # print(data.rel_set[key])
        if int(key) == 3:
            print(data.rel_set[key])

get_similar_articles(query_array.pop(3),df)