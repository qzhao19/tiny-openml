import numpy as np
import operator
import math

def calc_entropy(samples):
    """Computer entropy, which is defied as the expected value of information, first we need to define inforamtion, if 
       we classify something that can take on mulitiple values, the information for Xi is: l(Xi)=log2 p(Xi), p(Xi) is prob
       of chossing this class. To calculate entropy, you need the expected value of all the information of all possible 
       values of our class.

       Args:
           data: a set of data which we want to use 
       Returns:

    """
    # calculate the number of sample
    num_sample = len(samples)
    label_counts = {}

    for sample in samples:
        label = sample[-1]
        if label not in label_counts.keys():
            label_counts[label] = 0
        else:
            label_counts[label] += 1
    entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key])/num_sample
        if prob == 0:
            entropy = 0
        else:
            entropy -= prob * math.log(prob, 2)
    return entropy


def split_dataset(samples, axis, value):
    """Split the dataset.  we want to mesure the entropy value of each feature throught current feature set. So that
    we need to split the current feature of dataset

    Args:
        samples: n-dimension matrix. the dataset we will split
        axis: int. the feature we'll split on
        value: the valur of the feature to return 
    Returns:
        splited dataset

    """
    splited_set = []
    # loop the dataset
    for sample in samples:
        # check if the feature value that we want to return
        if sample[axis] == value:
            reduced_feature = sample[:axis]
            reduced_feature.extend(sample[axis+1:])
            splited_set.append(reduced_feature)
    return splited_set



def choose_best_feature(samples):
    """This function is performed to choose to features, split on the dataset and organize the best splited feature 
    dataset.We’ve made a few assumptions about the data. The first assumption is that it comes in the form of a list 
    of lists, and all these lists are of equal size. The next assumption is that the last column in the data or the 
    last item in each instance is the class label of that instance. 

    Args:
        samples: n-dimensions, dataset

    """

    num_features = len(samples[0]) - 1
    best_entropy = calc_entropy(samples)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        # create a unique list of class labels 
        features = [sample[i] for sample in samples]
        unique_features = set(features)
        entropy = 0.0
        for unique_feature in unique_features:
            # calculate the value of entropy for each split
            sub_samples = split_dataset(samples, i, unique_feature)
            prob = len(sub_samples)/float(len(samples))
            entropy += prob * calc_entropy(sub_samples)
        info_gain = best_entropy - entropy
        # find the best information gain
        if (info_gain > best_info_gain):
            best_info_gain=info_gain
            best_feature = i
    return best_feature
    
def max_vote(classes):
    """

    """
    class_count = {}
    for vote in classes:
        if vote not in class_count.key():
            class_count[vote] = 0
        else:
            class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_classes_count[0][0]

def create_tree(samples, labels):
    """

    """
    classes = [sample[-1] for sample in samples]
    if classes.count(classes[0]) == len(classes):
        return classes[0]
    if len(samples[0]) == 1:
        return max_vote(classes)

    best_feature = choose_best_feature(samples)
    # print(best_feature)
    # print(labels[0])
    best_feature_label = labels[best_feature]
    tree = {best_feature_label:{}}
    del(labels[best_feature])
    feature_value = [sample[best_feature] for sample in samples]
    values = set(feature_value)
    for value in values:
        sub_labels = labels[:]
        sub_samples = split_dataset(samples, best_feature, value)
        tree[best_feature_label][value] = create_tree(sub_samples, sub_labels)
    return tree
