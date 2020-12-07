class Evaluation:
    def __int__(self):
        pass
    def evaluate_BM25(self,relevant_docs_qId,query_result_qId):
        # print(relevant_docs_qId)
        #before calculating the relevance set get feedbacks on the relevance information
        #you can generate random feedback information
        true_positives = 0
        false_positives = 0
        #query result is a dictinary I am iterating for keys only
        precision_array = [] #each step calculate precision it is needed only for to show precision at 5 or 20
        interpolated_precesion = [] # take the max precision from current position to onwards
        avg_precision = [] # take the avg upto current position and return
        cumulative_gain = 0 #adding all precision values for the query and return
        for doc_id in query_result_qId.keys():
            if doc_id in relevant_docs_qId:  # position 3 indicates document ID
                true_positives += 1
            else:
                false_positives += 1
        recall = float(true_positives) / float(len(relevant_docs_qId))
        relevant_items_retrieved=true_positives+false_positives
        precision=float(true_positives)/float(relevant_items_retrieved)

        # compute total precision and recall
        print(" Precision: ",precision)
        print(" Recall:  ",recall)
        # true_positives = 0
        # false_positives = 0
        # recall = []
        # precision = []
        # for doc in ranking:
        #     if str(doc[0]) in relevants_docs_query[query_id]:  # position 3 indicates document ID
        #         true_positives += 1
        #     else:
        #         false_positives += 1
        #
        #     recall.append(self.get_recall(true_positives, len(relevants_docs_query[query_id])))
        #     precision.append(self.get_precision(true_positives, false_positives))
        #
        # recalls_levels = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
        #
        # interpolated_precisions = self.interpolate_precisions(recall, precision, recalls_levels)
        # self.plot_results(recalls_levels, interpolated_precisions)
        # plot.show()
        # plot.close()

    def evaluate_tf_idf_MS(self,relevant_docs_qId, query_result_qId):
        # print(relevant_docs_qId)
        true_positives = 0
        false_positives = 0
        # query result is a dictinary I am iterating for keys only
        for doc_id in query_result_qId.keys():

            if str(doc_id) in relevant_docs_qId:  # position 3 indicates document ID
                true_positives += 1
            else:
                false_positives += 1
        recall = float(true_positives) / float(len(relevant_docs_qId))
        relevant_items_retrieved = true_positives + false_positives
        precision = float(true_positives) / float(relevant_items_retrieved)

        # compute total precision and recall
        print(" Precision: ",precision)
        print(" Recall:  ",recall)

    def evaluate_tf_idf_cosine(self,relevant_docs_qId, query_result_qId):
        # print(relevant_docs_qId)
        true_positives = 0
        false_positives = 0
        # query result is a dictinary I am iterating for keys only
        for doc_id in query_result_qId.keys():

            if str(doc_id) in relevant_docs_qId:  # position 3 indicates document ID
                true_positives += 1
            else:
                false_positives += 1
        recall = float(true_positives) / float(len(relevant_docs_qId))
        relevant_items_retrieved = true_positives + false_positives
        precision = float(true_positives) / float(relevant_items_retrieved)

        # compute total precision and recall
        print(" Precision: " ,precision)
        print(" Recall:  " ,recall)
