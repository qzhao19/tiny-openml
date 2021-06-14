#ifndef SVM_DATA_H_
#define SVM_DATA_H__

#include "svm_node.h"


struct svm_data {
    /*store the dataset that involved the calculate, there are 
    the number of samples, *lable is an array what the lable to 
    which the sample belongs, **data: an 2D array where the 
    content is the pointer*/

    int n_samples;
    double *label;
    struct svm_node **data;
};



#endif //SVM_DATA_H__//