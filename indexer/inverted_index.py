#Date: 12/09/2020
#Class: CS6821
#Project: A Information Retrieval System
#Author(s): Mohammad Jaminur Islam
class Inverted_Index:
    def __init__(self):
        self.inverted_index = dict()

    def __contains__(self, item):
        return item in self.inverted_index

    def __getitem__(self, item):
        return self.inverted_index[item]

    def map_word_with_document(self, word, doc_id):
        # print("word: ",word," doc_id: ",doc_id)
        if word in self.inverted_index:
            if doc_id in self.inverted_index[word]:
                self.inverted_index[word][doc_id] += 1
            else:
                self.inverted_index[word][doc_id] = 1
        else:
            document_freq = dict()
            document_freq[doc_id] = 1
            self.inverted_index[word] = document_freq

        # frequency of word in document

    def get_document_frequency(self, word, doc_id):
        if word in self.inverted_index:
            if doc_id in self.inverted_index[word]:
                return self.inverted_index[word][doc_id]
            else:
                print('%s not in document %s' % (str(word), str(doc_id)))
                # raise LookupError('%s not in document %s' % (str(word), str(doc_id)))
        else:
            # raise LookupError('%s not in inverted_index' % str(word))
            print('%s not in inverted_index' % str(word))

    def get_index_frequency(self, word):
        if word in self.inverted_index:
            return len(self.inverted_index[word])
        else:
            print('%s not in inverted_index' % word)
            # raise LookupError('%s not in inverted_index' % word)

class Document_Length_Table:
    def __init__(self):
        self.document_table = dict()

    def map_document_and_length(self, doc_id, length):
        self.document_table[doc_id] = length

    def get_length(self, doc_id):
        if doc_id in self.document_table:
            return self.document_table[doc_id]
        else:
            print('%s not found in table' % str(doc_id))

    def get_average_length(self):
        sum = 0
        for length in self.document_table.values():
            sum += length
        return float(sum) / float(len(self.document_table))

