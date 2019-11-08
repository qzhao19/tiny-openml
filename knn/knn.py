import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    """Download dating data to testing alogrithm
    Args:
        file_name: input file path which allows to read a txt file
    Returns:
        retuen_mat: a matrix of dating data contains 3 attributs: 
                        1. Number of frequent flyer miles earned per year
                        2. Percentage of time spent playing video games
                        3. Liters of ice cream consumed per week
        label_vect: a vectro conatins labels 
    """
    with open(file_name, 'r') as file:
        contents = [line.strip().split('\t') for line in file.readlines()]

    label_map = {'largeDoses': '3', 'smallDoses': '2', 'didntLike': '1'}
    feature_list = []
    label_list = []
    for i in range(len(contents)):
        feature_list.append(contents[i][:3])
        label_list.append(contents[i][-1])
    
    for indices, value in enumerate(label_list):
        if not value.isdigit():
            if value in label_map.keys():
                label_list[indices] = label_map[value]
                
    return np.array(feature_list).astype(float), np.asarray(label_list).astype(int)


def plot2D(data, label):
    fig = plt.figure(figsize=(8, 6))
    colors = ('red', 'green', 'blue')
    groups = ('Did Not Like', 'Liked in Small Doses', 'Liked in Large Doses') 
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    ax = fig.add_subplot(111)
    for color, group in zip(colors, groups):
        ax.scatter(data[:,0], data[:,1], 
                   15.0*label, 15.0*label,
                   alpha=0.8, label=group)
    ax.legend(loc=2)
    plt.show()
# plot2D(features, labels)


def normalize(data):
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

def knn_classify(sample, data, label, k=5):
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
    for i in range(k):
        voted_label = label[sorted_dist_indices[i]]
        label_cnt[voted_label] = label_cnt.get(voted_label, 0) + 1
    sorted_label_cnt = sorted(label_cnt.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_cnt[0][0]

def knn_classify_test(data, label, ratio=0.4):
    norm_data = normalize(data)
    sample_nums, feature_nums = data.shape
    test_sample_nums = int(ratio * sample_nums)
    error_count = 0.0
    for i in range(test_sample_nums):
        classifier_result=knn_classify(norm_data[i,:], norm_data[test_sample_nums:sample_nums,:], labels[test_sample_nums:sample_nums])
        print('the classifier came back with: %d, the real answer is: %d' %(classifier_result, label[i]))
        if classifier_result != label[i]:
            error_count+=1.0
    print('the total error rate is: %f' % (error_count/float(test_sample_nums)))
    


