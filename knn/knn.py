import numpy as np
import matplotlib.pyplot as plt


class KNeighborsClassifier(object):
    """Classifier implementing the k-nearest neighbors vote.
    """
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def _normalize(self, data):
        """Normalize values that lie in differents ranges, commons ranges to normalisze them to are 0 to 1 or -1 to 1 newValue = (oldValue-min)/(max-min)
        Args:
            data: data that we want to normalizing
        Returns:
            norm_data: normalized data
        """
        sample_nums, feature_nums = data.shape
        min_value = data.min(0)
        max_value = data.max(0)
        range_value = max_value - min_value
        norm_data = np.zeros((sample_nums, feature_nums), dtype=data.dtype)
        norm_data = (data - np.tile(min_value, (sample_nums, 1))) / (np.tile(max_value, (sample_nums, 1)) - np.tile(min_value, (sample_nums, 1)))
        return norm_data

    def _knn_classify(self, data, label):
        """Function knnClassify realize pricipal KNN algorithm
        Args::
            sample: the input vector to classify
            data: full matrix of training examples
            labes: a vector of labels 
        Returns:
            the label of the item occurring the most frequently
        """

        sample_nums, feature_nums = data.shape
        dist_list = np.sqrt((np.tile(sample_nums, 1) - data)**2).sum(axis=1)
        sorted_dist_indices = np.argsort(dist_list)
        label_cnt = {}
        for i in range(self.n_neighbors):
            voted_label = label[sorted_dist_indices[i]]
            label_cnt[voted_label] = label_cnt.get(voted_label, 0) + 1
        sorted_label_cnt = sorted(label_cnt.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_label_cnt[0][0]

    def fit(self, data, label, ratio):

        norm_data = self._normalize(data)
        sample_nums, feature_nums = data.shape
        test_sample_nums = int(ratio * sample_nums)
        error_count = 0.0
        for i in range(test_sample_nums):
            classifier_result=self._knn_classify(norm_data[i,:], norm_data[test_sample_nums:sample_nums,:], labels[test_sample_nums:sample_nums])
            print('the classifier came back with: %d, the real answer is: %d' %(classifier_result, label[i]))
            if classifier_result != label[i]:
                error_count+=1.0
        print('the total error rate is: %f' % (error_count/float(test_sample_nums)))

