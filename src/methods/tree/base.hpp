#ifndef METHODS_TREE_BASE_HPP
#define METHODS_TREE_BASE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace decision_tree {

template<typename DataType>
struct Node{

    std::size_t num_samples_per_class;
    std::size_t num_samples;
    std::size_t feature_idx;
    DataType threshold;
    DataType value;
    bool is_leaf;

    Node* left_child;
    Node* right_child;

    Node(std::size_t num_samples_per_class_ = 0, 
        std::size_t num_samples_ = 0, 
        std::size_t feature_idx_ = 0, 
        DataType threshold_ = ConstType<DataType>::quiet_NaN(), 
        DataType value_ = ConstType<DataType>::quiet_NaN(), 
        bool is_leaf_ = false,
        Node* left_child_ = nullptr, 
        Node* right_child_ = nullptr): 
            num_samples_per_class(num_samples_per_class_),
            num_samples(num_samples_), 
            feature_idx(feature_idx_),
            threshold(threshold_),
            value(value_),
            is_leaf(is_leaf_),
            left_child(left_child_),
            right_child(right_child_) {};
};



template<typename DataType>
class DecisionTree {
private:
    std::size_t min_samples_split;
    std::size_t max_depth;
    double min_impurity;

protected:
    


public:
    DecisionTree(): min_samples_split(2), 
        max_depth(3), 
        min_impurity(1e7) {};

    DecisionTree(std::size_t min_samples_split_,
        std::size_t max_depth_,
        double min_impurity_): 
            min_samples_split(min_samples_split_), 
            max_depth(max_depth_), 
            min_impurity(min_impurity_) {};


    ~DecisionTree() {};





};

} // decision_tree
} // openml

#endif /*METHODS_TREE_BASE_HPP*/
