
    
        
class KNN(object):
    def __init__(self, k, test_rate):
        self.k=k
        self.test_rate=test_rate
        
    def normalizeData(self, data):
        """Normalize values that lie in differents ranges, commons ranges to normalisze them to are 0 to 1 or -1 to 1
           newValue = (oldValue-min)/(max-min)
           Paramsters:
               input_data: data that we want to normalizing
           Returns:
               norm_data: normalized data
        """
        # get min/max value from the colmuns, here should use min(0)
        min_val=data.min(0)
        max_val=data.max(0)
        ranges=max_val-min_val
        norm_data=np.zeros(data.shape, dtype=float)
        num_rows=data.shape[0]
        norm_data=(data-np.tile(min_val, (num_rows, 1)))/(np.tile(max_val, (num_rows, 1))-np.tile(min_val, (num_rows, 1)))
        return norm_data
    
    def classify(self, x, data, labels):
        """Function knnClassify realize pricipal KNN algorithm
           Parameters:
               x: the input vector to classify
               data: full matrix of training examples
               labes: a vector of labels 
           Returns:
               the label of the item occurring the most frequently
        """
        num_rows, num_cols=data.shape
        # caculate the distance between input classify vector and training matrix 
        distance=np.sqrt(((np.tile(x, (num_rows,1))-data)**2).sum(axis=1))
        # print(distance)
        # sort distance and get their sorted index 
        sortd_dist_indice=np.argsort(distance)
        class_count={}
        for i in range(self.k):
            voted_lables=labels[sortd_dist_indice[i]]
            class_count[voted_lables]=class_count.get(voted_lables,0)+1
        sorted_class_count=sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]
        
    def knnClassifyTest(self, data, labels):
        norm_data=self.normalizeData(data)
        num_rows, num_cols=data.shape
        num_test=int(self.test_rate*num_rows)
        error_count=0.0
        for i in range(num_test):
            classifier_result=self.classify(norm_data[i,:], norm_data[num_test:num_rows,:], labels[num_test:num_rows])
            print('the classifier came back with: %d, the real answer is: %d' %(classifier_result, labels[i]))
            if classifier_result!=labels[i]:
                error_count+=1.0
        print('the total error rate is: %f' % (error_count/float(num_test)))
