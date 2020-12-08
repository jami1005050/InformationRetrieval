#Date: 12/09/2020
#Class: CS6821
#Project: A Information Retrieval System
#Author(s): Mohammad Jaminur Islam
import matplotlib.pyplot as plot
import numpy as np
class Evaluation:
    def __int__(self):
        pass

    def relevant_doc_retrieved(self, query, ranking, relevants_docs_query):
        true_positives = 0
        false_positives = 0
        for doc in ranking:
            if str(doc[0]) in relevants_docs_query[query]:  # position 3 indicates document ID
                true_positives += 1
            else:
                false_positives += 1
        return true_positives, false_positives

    def get_recall(self, true_positives, real_true_positives):
        recall = float(true_positives) / float(real_true_positives)
        return recall

    def get_precision(self, true_positives, false_positives):
        relevant_items_retrieved = true_positives + false_positives
        precision = float(true_positives) / float(relevant_items_retrieved)
        return precision

    def interpolate_precisions(self, recalls, precisions, recalls_levels):
        precisions_interpolated = np.zeros((len(recalls), len(recalls_levels)))
        i = 0
        while i < len(precisions):
            # use the max precision obtained for the topic for any actual recall level greater than or equal the recall_levels
            recalls_inter = np.where((recalls[i] > recalls_levels) == True)[0]
            for recall_id in recalls_inter:
                if precisions[i] > precisions_interpolated[i, recall_id]:
                    precisions_interpolated[i, recall_id] = precisions[i]
            i += 1

        mean_interpolated_precisions = np.mean(precisions_interpolated, axis=0)
        return mean_interpolated_precisions

    def plot_results(self, recall, precision, modelname):
        plot.plot(recall, precision)
        plot.xlabel('recall')
        plot.ylabel('precision')
        plot.draw()
        plot.title('P/R curves for'+modelname)
    def evaluate_BM25(self,relevant_docs_qId,query_result_qId):
        true_positives = 0
        false_positives = 0
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
        true_positives = 0
        false_positives = 0
        recall = []
        precision = []
        for  doc_id in query_result_qId.keys():
            if doc_id in relevant_docs_qId:   # position 3 indicates document ID
                true_positives += 1
            else:
                false_positives += 1

            recall.append(self.get_recall(true_positives, len(relevant_docs_qId)))
            precision.append(self.get_precision(true_positives, false_positives))

        recalls_levels = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
        print("precision")
        print(precision)
        for row in precision:
            cumulative_gain+= row
        print("Cumulative Gain for BM25 model is:",cumulative_gain)
        interpolated_precisions = self.interpolate_precisions(recall, precision, recalls_levels)
        self.plot_results(recalls_levels, interpolated_precisions,"BM25")
        plot.show()
        plot.close()

    def evaluate_tf_idf_MS(self,relevant_docs_qId, query_result_qId):
        print(relevant_docs_qId)
        true_positives = 0
        false_positives = 0
        cumulative_gain = 0 #adding all precision values for the query and return
        print(query_result_qId)
        for doc_id in query_result_qId.keys():
            if str(doc_id) in relevant_docs_qId:  # position 3 indicates document ID
                true_positives += 1
                print("Matched Document:",doc_id)
            else:
                false_positives += 1
        recall = float(true_positives) / float(len(relevant_docs_qId))
        relevant_items_retrieved = true_positives + false_positives
        precision = float(true_positives) / float(relevant_items_retrieved)
        print(" Precision: ",precision)
        print(" Recall:  ",recall)
        true_positives = 0
        false_positives = 0
        recall = []
        precision = []
        for doc_id in query_result_qId.keys():
            if str(doc_id) in relevant_docs_qId:  # position 3 indicates document ID
                true_positives += 1
            else:
                false_positives += 1
            recall.append(self.get_recall(true_positives, len(relevant_docs_qId)))
            precision.append(self.get_precision(true_positives, false_positives))
        recalls_levels = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
        print("precision")
        print(precision)
        for row in precision:
            cumulative_gain += row
        print("Cumulative Gain for Maching Score model is:", cumulative_gain)
        interpolated_precisions = self.interpolate_precisions(recall, precision, recalls_levels)
        self.plot_results(recalls_levels, interpolated_precisions, " Maching Score")
        plot.show()
        plot.close()

    def evaluate_tf_idf_cosine(self,relevant_docs_qId, query_result_qId):
        true_positives = 0
        false_positives = 0
        cumulative_gain = 0
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
        true_positives = 0
        false_positives = 0
        recall = []
        precision = []
        for doc_id in query_result_qId.keys():
            if str(doc_id) in relevant_docs_qId:  # position 3 indicates document ID
                true_positives += 1
            else:
                false_positives += 1
            recall.append(self.get_recall(true_positives, len(relevant_docs_qId)))
            precision.append(self.get_precision(true_positives, false_positives))
        recalls_levels = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
        print("precision")
        print(precision)
        for row in precision:
            cumulative_gain += row
        print("Cumulative Gain for Cosine Similarity model is:", cumulative_gain)
        interpolated_precisions = self.interpolate_precisions(recall, precision, recalls_levels)
        self.plot_results(recalls_levels, interpolated_precisions, " Cosine Similarity")
        plot.show()
        plot.close()
