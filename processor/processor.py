from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from num2words import num2words

#region removal of stop words
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
class Processor:
    def __init__(self):
        self.tokenized_doc_set = dict()
        self.tokenized_query_set = dict()

    def convert_lower_case(self,data):
        return np.char.lower(data)


    def remove_stop_words(self,data):
        stop_words = stopwords.words('english')
        words = word_tokenize(str(data))
        new_text = ""
        for w in words:
            if w not in stop_words and len(w) > 1:
                new_text = new_text + " " + w
        return new_text


    def remove_punctuation(self,data):
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        for i in range(len(symbols)):
            data = np.char.replace(data, symbols[i], ' ')
            data = np.char.replace(data, "  ", " ")
        data = np.char.replace(data, ',', '')
        return data


    def remove_apostrophe(self,data):
        return np.char.replace(data, "'", "")



    def stemming(self,data):
        stemmer = PorterStemmer()

        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + stemmer.stem(w)
        return new_text

    def convert_numbers(self,data):
        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            try:
                w = num2words(int(w))
            except:
                a = 0
            new_text = new_text + " " + w
        new_text = np.char.replace(new_text, "-", " ")
        return new_text

    def normal_tokenizing(self,doc_set):
        for doc_id in doc_set:
            word_tokens = word_tokenize(doc_set[doc_id])
            self.tokenized_doc_set[doc_id] = word_tokens
    def normal_q_toeknizing(self,query_set):
        for doc_id in query_set:
            word_tokens = word_tokenize(query_set[doc_id])
            self.tokenized_query_set[doc_id] = word_tokens
    def pre_normal_process(self,query):
        return word_tokenize(query)

    def process_document_set(self,doc_set):
        for doc_id in doc_set:
            word_tokens = self.pre_process(doc_set[doc_id])
            self.tokenized_doc_set[doc_id] = word_tokens

    def pre_process(self,query):
        data = self.convert_lower_case(query)
        data = self.remove_punctuation(data)  # remove comma seperately
        data = self.remove_apostrophe(data)
        data = self.remove_stop_words(data)
        data = self.convert_numbers(data)
        data = self.stemming(data)
        data = self.remove_punctuation(data)
        data = self.convert_numbers(data)
        data = self.stemming(data)  # needed again as we need to stem the words
        data = self.remove_punctuation(data)  # needed again as num2word is giving few hypens and commas fourty-one
        data = self.remove_stop_words(data)  # needed again as num2word is giving stop words 101 - one hundred and one
        word_tokens = word_tokenize(data)  # words with stopwords
        return word_tokens

    def process_query_set(self,query_set):
        for query_id in query_set:
            word_tokens = self.pre_process(query_set[query_id])
            self.tokenized_query_set[query_id] = word_tokens


#we can use below code as word tokenization
# pattern = r'\w+'
# regex_words = regexp_tokenize(example_sent, pattern)
# print("regex words:", regex_words)
# stropword_free_words = [w for w in word_tokens if not w in stop_words] #words without stepwords and withour stemming
# #region stemming
# stemmed_words = [] # list of stemming words
# for w in stropword_free_words:
#     stemmed_words.append(ps.stem(w))
# print(stemmed_words)
#
# unique_words_for_doc = list(dict.fromkeys(stemmed_words))
# print(unique_words_for_doc)
# break
#
# #Term frequency
# def termFrequency(term, doc):
#     """
#     Input: term: Term in the Document, doc: Document
#     Return: Normalized tf: Number of times term occurs
#       in document/Total number of terms in the document
#     """
#     # Splitting the document into individual terms
#     normalizeTermFreq = doc.lower().split()
#
#     # Number of times the term occurs in the document
#     term_in_document = normalizeTermFreq.count(term.lower())
#
#     # Total number of terms in the document
#     len_of_document = float(len(normalizeTermFreq))
#
#     # Normalized Term Frequency
#     normalized_tf = term_in_document / len_of_document
#
#     return normalized_tf
#
#
# def inverseDocumentFrequency(term, allDocs):
#     num_docs_with_given_term = 0
#
#     """
#     Input: term: Term in the Document,
#            allDocs: List of all documents
#     Return: Inverse Document Frequency (idf) for term
#             = Logarithm ((Total Number of Documents) /
#             (Number of documents containing the term))
#     """
#     # Iterate through all the documents
#     for doc in allDocs:
#
#         """
#         Putting a check if a term appears in a document.
#         If term is present in the document, then
#         increment "num_docs_with_given_term" variable
#         """
#         if term.lower() in allDocs[doc].lower().split():
#             num_docs_with_given_term += 1
#
#     if num_docs_with_given_term > 0:
#         # Total number of documents
#         total_num_docs = len(allDocs)
#
#         # Calculating the IDF
#         idf_val = log(float(total_num_docs) / num_docs_with_given_term)
#         return idf_val
#     else:
#         return 0
