#Date: 12/09/2020
#Class: CS6821
#Project: A Information Retrieval System
#Author(s): Mohammad Jaminur Islam
import math
import operator
from math import log
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from data_loader.data import Data
from evaluation.evaluation import Evaluation
from indexer.inverted_index import Inverted_Index, Document_Length_Table
from processor.processor import Processor
from ranking.ranking import score_BM25
from collections import Counter
THRESHOLD = 50 #top 50 high ranked documents for a method
vectorizer = TfidfVectorizer()

def inverseDocumentFrequency(term, doc_set):
    num_docs_with_given_term = 0
    """ 
    Input: term: Term in the Document, 
           allDocs: List of all documents 
    Return: Inverse Document Frequency (idf) for term 
            = Logarithm ((Total Number of Documents) /  
            (Number of documents containing the term)) 
    """
    # Iterate through all the documents
    for doc_id in doc_set:
        """ 
        Putting a check if a term appears in a document. 
        If term is present in the document, then  
        increment "num_docs_with_given_term" variable 
        """
        if term.lower() in doc_set[doc_id]:
            num_docs_with_given_term += 1

    if num_docs_with_given_term > 0:
        # Total number of documents
        total_num_docs = len(doc_set)

        # Calculating the IDF
        idf_val = log(float(total_num_docs) / num_docs_with_given_term)
        return idf_val
    else:
        return 0

def termFrequency(term, doc):
    """
    Input: term: Term in the Document, doc: Document
    Return: Normalized tf: Number of times term occurs
      in document/Total number of terms in the document
    """
    # Splitting the document into individual terms
    # Number of times the term occurs in the document
    print(term)
    term_in_document = doc.count(term.lower())
    print("count",term_in_document)
    # Total number of terms in the document
    len_of_document = float(len(doc))
    # Normalized Term Frequency
    normalized_tf = term_in_document / len_of_document
    return normalized_tf

def build_data_structures(tokenized_doc_set):
    print("Building Data Structure When Advance Tokenizing is used")
    idx = Inverted_Index()
    dlt = Document_Length_Table()
    term_frec = dict()
    inverse_doc_frec = dict()
    tf_idf = dict()
    for doc_id in tokenized_doc_set:
        # print(tokenized_doc_set[doc_id])
        # build inverted inverted_index
        # print("\n")
        for word in tokenized_doc_set[doc_id]:
            # print(word)
            idx.map_word_with_document(str(word), str(doc_id))
            # #region term frequency for a document
            # if not (word,doc_id) in term_frec:
            #     term_frec[word,doc_id] = termFrequency(word,tokenized_doc_set[doc_id])
            # if not (word,doc_id) in inverse_doc_frec:
            #     inverse_doc_frec[word,doc_id] = inverseDocumentFrequency(word,tokenized_doc_set)
            # if not (word,doc_id) in tf_idf:
            #     tf_idf[word,doc_id] = term_frec[word,doc_id] * inverse_doc_frec[word,doc_id]
            # #endregion
        # build document length table
        length = len(tokenized_doc_set[str(doc_id)])
        dlt.map_document_and_length(doc_id, length)
    # #region saving tf_idf
    # term_frec_df = pd.Series(term_frec).reset_index()
    # term_frec_df.columns = ['word', 'doc_id', 'tf']
    # # term_frec_df.to_csv('term_freq.csv')
    # inverse_doc_frec_df = pd.Series(inverse_doc_frec).reset_index()
    # inverse_doc_frec_df.columns = ['word', 'doc_id', 'idf']
    # # inverse_doc_frec_df.to_csv('inv_doc_frq.csv')
    #
    # tf_idf_df = pd.Series(tf_idf).reset_index()
    # tf_idf_df.columns = ['word', 'doc_id', 'tf-idf']
    # # merged_df = pd.DataFrame()
    # merged_df = pd.merge(term_frec_df,inverse_doc_frec_df,on=['word','doc_id'])
    # merged_df_with_tf_idf = pd.merge(merged_df,tf_idf_df,on=['word','doc_id'])
    # merged_df_with_tf_idf.to_csv('tf_idf.csv')
    # print("Calculated Tf length",len(term_frec),"idf length",len(inverse_doc_frec)," tf_idf length:",len(tf_idf) )
    # #endregion
    return idx, dlt

#region Ranking through bm25
def process_query_bm25(query_tokenize, index_list, doc_lookup_table):
    query_result = dict()
    for term in query_tokenize:
        # print(term)
        if term in index_list:
            doc_dict = index_list[term]  # retrieve index entry
            # print("Document dictionary")
            # print(doc_dict)
            for docid, freq in doc_dict.items():  # for each document and its word frequency
                score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(doc_lookup_table.document_table),
                                   dl=doc_lookup_table.get_length(docid),
                                   avdl=doc_lookup_table.get_average_length())  # calculate score
                if docid in query_result:  # this document has already been scored once
                    query_result[docid] += score
                else:
                    query_result[docid] = score
    # print("Ranking for BM25")
    top_document = dict()
    if(len(query_result)>THRESHOLD):
        # print("Show Top: ", THRESHOLD, " Documents for the query")
        sorted_x = sorted(query_result.items(), key=operator.itemgetter(1))
        sorted_x.reverse()
        # index = 0
        # print(sorted_x)
        for i in sorted_x[:THRESHOLD]: # showing top 20 results
            # tmp = {"doc_id":i[0], "serial":index, "rank":i[1]}
            top_document[i[0]] = i[1]
            # print('{:>1}\tQ0\t{:>4}\t{:>2}\t{:>12}\tNH-BM25'.format(*tmp))
            # index += 1
        print("Top ",THRESHOLD," Document ranking")
        print(top_document)
        return top_document
    print("All retrieved documents ranking")
    print(query_result)
    return query_result
#endregion
def matching_score(tokens,tf_idf):
    # preprocessed_query = preprocess(query)
    # tokens = word_tokenize(str(query))
    print("Matching Score")
    # print("\nQuery:", query)
    # print("")
    # print(tokens)
    query_weights = dict()
    for row in tf_idf.itertuples():
        if row.word in tokens:
            try:
                query_weights[row.doc_id] += row.tf_idf
            except:
                query_weights.setdefault(row.doc_id, row.tf_idf)
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)
    top_document = dict()
    if (len(query_weights) > THRESHOLD):
        # print("Show Top: ", THRESHOLD, " Documents for the query")
        # index = 0
        # print(sorted_x)
        for i in query_weights[:THRESHOLD]:  # showing top 20 results
            # tmp = {"doc_id":i[0], "serial":index, "rank":i[1]}
            top_document[i[0]] = i[1]
            # print('{:>1}\tQ0\t{:>4}\t{:>2}\t{:>12}\tNH-BM25'.format(*tmp))
            # index += 1
        print("Top ", THRESHOLD, " Document ranking")
        print(top_document)
        return top_document
    else:
        for i in query_weights[:len(query_weights)]:  # showing top 20 results
            # tmp = {"doc_id":i[0], "serial":index, "rank":i[1]}
            top_document[i[0]] = i[1]
        print("All retrieved documents ranking")
        print("Query Weight")
        print(top_document)
        return top_document
    # m_score_ary_for_doc = []
    # for i in query_weights[:k]:
    #     m_score_ary_for_doc.append(i[0])
    # print(m_score_ary_for_doc)

def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

def calculate_cosine_similarity(tokens, inverse_index,dlt,tf_idf):
    # print("query:", query)
    query_vector = getQueryVector(tokens,inverse_index,dlt)  #using sklearn library to get the query tfidf vector
    #can not use transfor without fit transform on document
    # print("Query Vectors")
    # print(query_vector)
    # query_vector = getQueryVector(query,tf_idf_frame) # return only the query elements tf-idf values based on documents
    document_vectors = get_doc_vector(inverse_index,tf_idf,dlt)
    # print(len(document_vectors))
    # print("Document Vectors")
    # print(document_vectors)
    d_cosines = []
    for d_vector in document_vectors:
        d_cosines.append(cosine_sim(query_vector, d_vector))
    # print("Similarities in the doc set")
    # print(d_cosines)
    ranking = dict()
    for item in d_cosines:
        if(item!=0):
            ranking.setdefault(d_cosines.index(item)+1,item)
    ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

    top_document = dict()
    if (len(ranking) > THRESHOLD):
        # print("Show Top: ", THRESHOLD, " Documents for the query")
        # index = 0
        # print(sorted_x)
        for i in ranking[:THRESHOLD]:  # showing top 20 results
            # tmp = {"doc_id":i[0], "serial":index, "rank":i[1]}
            top_document[i[0]] = i[1]
            # print('{:>1}\tQ0\t{:>4}\t{:>2}\t{:>12}\tNH-BM25'.format(*tmp))
            # index += 1
        print("Top ", THRESHOLD, " Document ranking")
        print(top_document)
        return top_document
    print("All retrieved documents ranking")
    print("Length of vectors",len(ranking))
    print(ranking)
    return ranking
    # out = np.array(d_cosines).argsort()[-k:][::-1]
    # print("Top 10 documents among all ranked documents by cosine similarity for the query:")
    # print(out)

#this method will calculate the term frequency inverse document frequency
#used the documents to get the tf_idf for query tokens
#inverse_index and query dlt  used to create the query vectors of same shape
def getQueryVector(tokens, inverse_index,dlt):
    voca_ary = [k  for  k in  inverse_index]
    query_vector = np.zeros((len(inverse_index)))
    # tokens = word_tokenize(str(query))
    counter = Counter(tokens)
    words_count = len(tokens)
    for token in np.unique(tokens):
        tf = counter[token] / words_count
        df = inverse_index.get(token)
        df_len = 0
        if(df is not None):
            # print("document Frequency:",df)
            df_len = len(df)
        idf = math.log((len(dlt) + 1) / (df_len + 1))
        try:
            ind = voca_ary.index(token)
            query_vector[ind] = tf * idf
        except:
            pass
    return query_vector

#the below method will return the vectors for each of the documents
#inverse index is used to get total number of words available in the document
def get_doc_vector(inverse_index,tf_idf,dlt): #dlt is document table lookup only used to track number of documents
    document_vectors = np.zeros((len(dlt), len(inverse_index)))
    voca_ary = [k  for  k in  inverse_index]
    for row in tf_idf.itertuples():
        try:
            ind = voca_ary.index(row.word)
            document_vectors[row.doc_id][ind] = row.tf_idf
        except:
            pass
    return document_vectors

def display_result(query_result):
    # print(query_result)
    # for result in query_result:
    sorted_x = sorted(query_result.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    # index = 0
    print(sorted_x)
    # for i in sorted_x[:20]: # showing top 20 results
    #     tmp = {"doc_id":i[0], "serial":index, "rank":i[1]}
    #     print(tmp)
    #     # print('{:>1}\tQ0\t{:>4}\t{:>2}\t{:>12}\tNH-BM25'.format(*tmp))
    #     index += 1

def main():
    print("Please wait.... Statring")
    data = Data()
    data.readData()
    evaluator = Evaluation()
    # print(len(data.doc_set))
    # print("Frame size:",len(tf_idf_frame))
    print("1. Stemming and Stopword removal while tokenizing")
    print("2. Normal Tokenizing without stopword and stemming")
    print("Select 1/2 for above ")
    type_toke = input()
    processor = Processor()
    if(type_toke == str(1)):
        tf_idf_frame = pd.read_csv('tf_idf.csv')
        processor.process_document_set(data.doc_set)
        processor.process_query_set(data.qry_set)
    else:
        tf_idf_frame = pd.read_csv('tf_idf_normal_tokenized.csv')
        processor.normal_tokenizing(data.doc_set)
        processor.normal_q_toeknizing(data.qry_set)
    # print(data.doc_set[int(1)])
    # print("Tokenized document set")
    # print(processor.tokenized_doc_set)
    idx, dlt = build_data_structures(processor.tokenized_doc_set)
    # print("idx")
    # print(idx.inverted_index)
    # print("dlt")
    # print(len(dlt.document_table))
    print("****Search engine Started******")
    while(True):
        print("Want to search? y/n")
        option = input()
        if option == "y" or option.lower()=="y":
            print("Enter Query:")
            query = input()
            query_tokenize = None
            if(type_toke == str(1)):
                query_tokenize = processor.pre_process(query)
            else:
                query_tokenize = processor.pre_normal_process(query)
            print("1.BM25 Model Ranking.\n 2. Tf_Idf Maching Score.\n 3. Tf_Idf Cosine Similarity\n")
            print("Select a option. 1/2/3")
            selection = input()
            if((selection) == str(1)): #1 for BM25 model Selection
                query_result = process_query_bm25(query_tokenize, idx.inverted_index, dlt)
                # display_result(query_result)
                print("Want to see the performance of the model:y/n")
                is_evaluate = input()
                if (is_evaluate == "y" or is_evaluate.lower() == "y"):
                    print("Using 'CISI.REL' for testing performance. It contains relevance information for queries in 'CISI.QRY'")
                    print("The query File has 112 queries and It can be accessed through ID from 1 to 112")
                    print("Select a Query ID 1-112 to test the Peformance of the Model comparing actual relevance")
                    query_id = input()
                    query_by_id = processor.tokenized_query_set[query_id]
                    # print("Query for Id:",query_id)
                    print("Query: ",data.qry_set[query_id])
                    relevant_docs_qId = data.rel_set[query_id]
                    print("relevant Documents: ")
                    print(relevant_docs_qId)
                    query_result_qId = process_query_bm25(query_by_id, idx.inverted_index, dlt)
                    evaluator.evaluate_BM25(relevant_docs_qId,query_result_qId)
            elif((selection)==str(2)): #2 for tf_idf matching score ranking model
                query_ranking = matching_score(query_tokenize,tf_idf_frame) #10 = the number of ranking we are seeking can be any number
                print("Want to see the performance of the model:y/n")
                is_evaluate = input()
                if (is_evaluate == "y" or is_evaluate.lower() == "y"):
                    print(
                        "Using 'CISI.REL' for testing performance. It contains relevance information for queries in 'CISI.QRY'")
                    print("The query File has 112 queries and It can be accessed through ID from 1 to 112")
                    print("Select a Query ID 1-112 to test the Peformance of the Model comparing actual relevance")
                    query_id = input()
                    query_by_id = processor.tokenized_query_set[query_id]
                    # print("Query for Id:",query_id)
                    print("Query: ", data.qry_set[query_id])
                    relevant_docs_qId = data.rel_set[query_id]
                    print("relevant Documents: ")
                    print(relevant_docs_qId)
                    query_result_qId =  matching_score(query_by_id,tf_idf_frame)
                    evaluator.evaluate_tf_idf_MS(relevant_docs_qId, query_result_qId)
            elif ((selection) == str(3)): #3 for tf_idf cosine similarity ranking model
                query_similarity = calculate_cosine_similarity(query_tokenize, idx.inverted_index, dlt.document_table,tf_idf_frame)  # 10 = the number of ranking we are seeking can be any number
                print("Want to see the performance of the model:y/n")
                is_evaluate = input()
                if (is_evaluate == "y" or is_evaluate.lower() == "y"):
                    print(
                        "Using 'CISI.REL' for testing performance. It contains relevance information for queries in 'CISI.QRY'")
                    print("The query File has 112 queries and It can be accessed through ID from 1 to 111")
                    print("Select a Query ID 1-111 to test the Peformance of the Model comparing actual relevance")
                    query_id = input()
                    query_by_id = processor.tokenized_query_set[query_id]
                    print("Query for Id:",query_id)
                    print("Query: ", data.qry_set[query_id])
                    relevant_docs_qId = data.rel_set[query_id]
                    print("relevant Documents: ")
                    print(relevant_docs_qId)
                    query_result_qId = calculate_cosine_similarity(query_by_id, idx.inverted_index, dlt.document_table,tf_idf_frame)
                    evaluator.evaluate_tf_idf_cosine(relevant_docs_qId, query_result_qId)
        else:
            print("Thank you for using our search engine. Bye")
            break

main()