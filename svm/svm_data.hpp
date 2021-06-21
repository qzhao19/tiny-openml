#ifndef SVM_DATA_HPP_
#define SVM_DATA_HPP_

#include "svm_node.hpp"


struct svm_data {
    /*store the dataset that involved the calculate, there are 
    the number of samples, *lable is an array what the lable to 
    which the sample belongs, **data: an 2D array where the 
    content is the pointer*/

    int n_samples;
    double *y;
    struct svm_node **X;
};



#endif //SVM_DATA_H__//
