#ifndef METHODS_TREE_BASE_HPP
#define METHODS_TREE_BASE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace decision_tree {

template<typename DataType>
struct Node {
    bool is_leaf;
    std::size_t num_samples_per_class;
    std::size_t num_samples;
    std::size_t feature_index;
    DataType threshold;
    DataType value;
    
    std::unique_ptr<Node> left_child;
    std::unique_ptr<Node> right_child;

    Node():is_leaf(false),
        num_samples_per_class(0),
        num_samples(0), 
        feature_index(ConstType<std::size_t>::max()),
        threshold(ConstType<DataType>::quiet_NaN()),
        value(ConstType<DataType>::quiet_NaN()),
        left_child(std::unique_ptr<Node>()), 
        right_child(std::unique_ptr<Node>()) {};

    
    explicit Node(bool is_leaf_,
        std::size_t num_samples_per_class_ = 0, 
        std::size_t num_samples_ = 0, 
        std::size_t feature_index_ = ConstType<std::size_t>::max(), 
        DataType threshold_ = ConstType<DataType>::quiet_NaN(), 
        DataType value_ = ConstType<DataType>::quiet_NaN()): 
            is_leaf(is_leaf_),
            num_samples_per_class(num_samples_per_class_),
            num_samples(num_samples_), 
            feature_index(feature_index_),
            threshold(threshold_),
            value(value_),
            left_child(std::unique_ptr<Node>()), 
            right_child(std::unique_ptr<Node>()){};

    ~Node(){};
};



template<typename DataType>
class DecisionTree {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    using NodeType = Node<DataType>;
    


    std::size_t min_samples_split;
    std::size_t max_depth;
    double min_impurity;


    struct SplitRecords {
        /* data */
        double largest_impurity;
        std::size_t best_feature_index;
        DataType best_feature_value; 
        MatType best_left_X;
        VecType best_left_y;
        MatType best_right_X;
        Vectype best_right_y;
    };
    
    std::unique_ptr<NodeType> root;


protected:
    template<typename ImpurityType>
    


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
