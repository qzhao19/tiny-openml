#ifndef METHODS_TREE_DECISION_TREE_CLASSIFIER_HPP
#define METHODS_TREE_DECISION_TREE_CLASSIFIER_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "base.hpp"
using namespace openml;

namespace openml {
namespace decision_tree {

template<typename DataType>
class DecisionTreeRegressor : public DecisionTree<DataType> {

private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    
    struct Node {
        bool is_leaf;
        double impurity;
        std::size_t num_samples;
        std::size_t feature_index;
        DataType feature_value;
        DataType predict_value;
        std::shared_ptr<Node> left_child;
        std::shared_ptr<Node> right_child;

        Node():is_leaf(true),
            impurity(0.0),
            num_samples(0),
            feature_index(ConstType<std::size_t>::max()),
            feature_value(ConstType<DataType>::quiet_NaN()),
            predict_value(ConstType<DataType>::quiet_NaN()),
            left_child(std::shared_ptr<Node>()), 
            right_child(std::shared_ptr<Node>()) {};

        explicit Node(bool is_leaf_,
            double impurity_ = 0.0,
            std::size_t num_samples_ = 0,
            std::size_t feature_index_ = ConstType<std::size_t>::max(), 
            DataType feature_value_ = ConstType<DataType>::quiet_NaN(), 
            DataType predict_value_ = ConstType<DataType>::quiet_NaN()): 
                is_leaf(is_leaf_),
                impurity(impurity_),
                num_samples(num_samples_), 
                feature_index(feature_index_),
                feature_value(feature_value_),
                predict_value(predict_value_),
                left_child(std::shared_ptr<Node>()), 
                right_child(std::shared_ptr<Node>()) {};

        ~Node(){};
    };
    std::shared_ptr<Node> root_;

protected:
    std::string criterion_;
    std::size_t min_samples_split_;
    std::size_t min_samples_leaf_;
    std::size_t max_depth_; 
    double min_impurity_decrease_;
    double stdev_;
    
    const double compute_impurity(const VecType& y, 
        const VecType& left_y, 
        const VecType& right_y) const {
        
        std::size_t num_samples = y.rows();
        std::size_t left_num_samples = left_y.rows();
        std::size_t right_num_samples = right_y.rows();


        if (criterion_ == "squared_error") {
            
            auto total_var = math::var<MatType, VecType>(y, -1);
            auto left_var = math::var<MatType, VecType>(left_y, -1);
            auto right_var = math::var<MatType, VecType>(right_y, -1);

            double mse = static_cast<double>(total_var.value()) - 
                static_cast<double>(left_num_samples) * static_cast<double>(left_var.value()) -
                    static_cast<double>(right_num_samples) * static_cast<double>(right_var.value());
            
            return mse;
        }
        else if (criterion_ == "absolute_error") {

            auto left_median = math::median<MatType, VecType>(left_y, -1);
            auto right_median = math::median<MatType, VecType>(right_y, -1);

        }

    }


public:
    DecisionTreeRegressor(): DecisionTree<DataType>(), 
        criterion_("squared_error"), 
        min_samples_split_(2), 
        min_samples_leaf_(0),
        max_depth_(4), 
        min_impurity_decrease_(1.0e-7),
        stdev_(1e-3) {};


    DecisionTreeRegressor(std::string criterion, 
        std::size_t min_samples_split, 
        std::size_t min_samples_leaf,
        std::size_t max_depth,
        double min_impurity_decrease,
        double stdev): DecisionTree<DataType>(), 
            criterion_(criterion), 
            min_samples_split_(min_samples_split), 
            min_samples_leaf_(min_samples_leaf),
            max_depth_(max_depth), 
            min_impurity_decrease_(min_impurity_decrease),
            stdev_(stdev) {};

};

} // decision_tree
} // openml

#endif /*METHODS_TREE_DECISION_TREE_CLASSIFIER_HPP*/
