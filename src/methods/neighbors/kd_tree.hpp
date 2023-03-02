#ifndef METHOD_NEIGHBORS_KD_TREE_HPP
#define METHOD_NEIGHBORS_KD_TREE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "./base.hpp"
using namespace openml;

namespace openml {
namespace neighbors {

template<typename DataType>
class KDTree {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    std::size_t p_;
    std::size_t num_neighbors_;
    std::string metric_;

    struct Node {
        bool is_leaf;
        double impurity;
        std::size_t num_samples;
        VecType num_samples_per_class;
        std::size_t feature_index;
        DataType feature_value;
        DataType predict_value;
        std::shared_ptr<Node> left_child;
        std::shared_ptr<Node> right_child;

        Node():is_leaf(true),
            impurity(0.0),
            num_samples(0),
            num_samples_per_class(VecType()),
            feature_index(ConstType<std::size_t>::max()),
            feature_value(ConstType<DataType>::quiet_NaN()),
            predict_value(ConstType<DataType>::quiet_NaN()),
            left_child(std::shared_ptr<Node>()), 
            right_child(std::shared_ptr<Node>()) {};

        explicit Node(bool is_leaf_,
            double impurity_ = 0.0,
            std::size_t num_samples_ = 0,
            VecType num_samples_per_class_ = VecType(), 
            std::size_t feature_index_ = ConstType<std::size_t>::max(), 
            DataType feature_value_ = ConstType<DataType>::quiet_NaN(), 
            DataType predict_value_ = ConstType<DataType>::quiet_NaN()): 
                is_leaf(is_leaf_),
                impurity(impurity_),
                num_samples(num_samples_), 
                num_samples_per_class(num_samples_per_class_),
                feature_index(feature_index_),
                feature_value(feature_value_),
                predict_value(predict_value_),
                left_child(std::shared_ptr<Node>()), 
                right_child(std::shared_ptr<Node>()) {};

        ~Node(){};
    };



public:
    KDTree(): num_neighbors_(15), 
        p_(1), 
        metric_("minkowski") {};
    

    KDTree(std::size_t num_neighbors, 
        std::size_t p, 
        std::string metric): num_neighbors_(num_neighbors), 
            p_(p), 
            metric_(metric) {};

};

} // neighbors
} // openml

#endif /*METHOD_NEIGHBORS_KD_TREE_HPP*/
