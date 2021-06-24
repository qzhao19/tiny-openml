#ifndef SVM_DATA_HPP_
#define SVM_DATA_HPP_

#include "svm_node.hpp"


struct svm_data {
    /*
     * Store all samples (data sets) that participated in the calculation this time, 
     * and their categories 
     * *y: is an array what the lable to which the sample belongs 
     * **data: an 2D array where the content is the pointer
     * 这样的数据结构有一个直接的好处，可以用x[i][j]来访问其中的某一元素
     */

    int n_samples;
    double *y;
    struct svm_node **X;
};



#endif //SVM_DATA_H__//
