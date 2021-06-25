#ifndef SVM_NODE_HPP_
#define SVM_NODE_HPP_

struct svm_node {
    /**
     * store the single feature in a single vector, for example, x1 = {0.5, 0.2, 0.6, 0.8}; so if we use 
     * svm_node to store it, which is a array what contains 5 svm_node, the memory map is:
     *       1      2      3     4     -1
     *       0.5    0.2    0.6   0.8    null
     * @params index: 存储单一向量的索引
     * @params value: 存储对应索引位置的值
    */
    int index;          
    double value;
};


#endif /*_SVM_NODE_H*/
