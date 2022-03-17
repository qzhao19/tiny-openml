#ifndef METHODS_NAIVE_BAYES_BASE_HPP
#define METHODS_NAIVE_BAYES_BASE_HPP
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

    Node* left_child;
    Node* right_child;

    Node(std::size_t num_samples_per_class_ = 0, 
        std::size_t num_samples_ = 0, 
        std::size_t feature_idx_ = 0, 
        DataType threshold_ = std::numeric_limits<DataType>::quiet_NaN(), 
        DataType value_ = std::numeric_limits<DataType>::quiet_NaN(), 
        Node* left_child_ = nullptr, 
        Node* right_child_ = nullptr): num_samples_per_class(num_samples_per_class_),
            num_samples(num_samples_), 
            feature_idx(feature_idx_),
            threshold(threshold_),
            value(value_),
            left_child(left_child_),
            right_child(right_child_)
};






} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_BASE_HPP*/
