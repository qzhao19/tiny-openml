#ifndef METHODS_TREE_DECISION_TREE_REGRESSOR_HPP
#define METHODS_TREE_DECISION_TREE_REGRESSOR_HPP
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

    /***/
    void delete_tree(std::shared_ptr<Node>& node) {
        if (node != nullptr) {
            delete_tree(node->left_child);
            delete_tree(node->right_child);
            std::move(node);
        }
    }

protected:
    std::string criterion_;
    std::size_t min_samples_split_;
    std::size_t min_samples_leaf_;
    std::size_t max_depth_; 
    double min_impurity_decrease_;
    double min_stdev_;
    
    const double compute_impurity(const VecType& y, 
        const VecType& left_y, 
        const VecType& right_y) const {
        
        std::size_t num_samples = y.rows();
        std::size_t left_num_samples = left_y.rows();
        std::size_t right_num_samples = right_y.rows();

        double impurity = 0.0;
        if (criterion_ == "squared_error") {
            auto total_var = math::var<MatType, VecType>(y, -1);
            auto left_var = math::var<MatType, VecType>(left_y, -1);
            auto right_var = math::var<MatType, VecType>(right_y, -1);

            double mse = static_cast<double>(total_var.value()) - 
                static_cast<double>(left_num_samples) * static_cast<double>(left_var.value()) -
                    static_cast<double>(right_num_samples) * static_cast<double>(right_var.value());
            
            impurity = mse;
        }
        else if (criterion_ == "absolute_error") {
            
            auto total_median = math::median<MatType, VecType>(y, -1);
            auto left_median = math::median<MatType, VecType>(left_y, -1);
            auto right_median = math::median<MatType, VecType>(right_y, -1);

            VecType y_bar = y.array() - total_median.array();
            VecType left_y_bar = left_y.array() - left_median.array();
            VecType right_y_bar = right_y.array() - right_median.array();

            auto y_bar_sum = math::sum<MatType, VecType>(y_bar, -1);
            auto left_y_bar_sum = math::sum<MatType, VecType>(left_y_bar, -1);
            auto right_y_bar_sum = math::sum<MatType, VecType>(right_y_bar, -1);

            double mae = static_cast<double>(y_bar_sum.value()) - 
                static_cast<double>(left_num_samples) * static_cast<double>(left_y_bar_sum.value()) -
                    static_cast<double>(right_num_samples) * static_cast<double>(right_y_bar_sum.value());

            impurity = mae;
        }
        else {
            throw std::invalid_argument("The criterion must be 'absolute_error' or 'squared_error'");
        }
        return impurity;
    }

    /**
     * 
    */
    const std::tuple<DataType, double> compute_node_value(
        const VecType& y) const {
        
        std::size_t num_samples = y.rows();

        auto y_bar = math::mean<MatType, VecType>(y, -1);
        VecType y_centered = y.array() - y_bar.value();

        double stdev_error = std::sqrt(
            static_cast<double>(y_centered.norm()) / static_cast<double>(num_samples)
        );

        return std::make_tuple(y_bar.value(), stdev_error); 
    }


    /** 
     * Build a decision tree by recursively finding the best split. 
    */
    void build_tree(const MatType& X, 
        const VecType& y, 
        std::shared_ptr<Node>& cur_node, 
        std::size_t cur_depth) const {
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        
        DataType predict_value;
        double min_stdev;
        
        std::tie(predict_value, min_stdev) = compute_node_value(y);

        cur_node->is_leaf = true;
        cur_node->num_samples = num_samples;
        cur_node->predict_value = predict_value;

        if (min_stdev <= min_stdev_) {
            return ;
        }
        
        // stopping split condition
        if (num_samples < min_samples_split_) {
            return ;
        }

        if (cur_depth > max_depth_) {
            return ;
        }

        std::size_t best_feature_index = ConstType<std::size_t>::max();
        DataType best_feature_value = ConstType<DataType>::quiet_NaN();
        double best_impurity = ConstType<double>::min();

        std::tie(best_impurity, best_feature_index, best_feature_value) = this->best_split(X, y);
        if (best_feature_index == ConstType<std::size_t>::max()) {
            return ;
        }

        if (best_impurity <= min_impurity_decrease_) {
            return ;
        }

        cur_node->impurity = best_impurity;
        cur_node->feature_index = best_feature_index;
        cur_node->feature_value = best_feature_value;
        // std::cout << "best_impurity = " << best_impurity << std::endl;

        VecType selected_x = X.col(best_feature_index);
        auto tmp1 = (selected_x.array() <= best_feature_value);
        VecType left_selected = tmp1.template cast<DataType>();
        IdxType left_selected_index = utils::where<VecType, IdxType>(left_selected);

        auto tmp2 = (selected_x.array() > best_feature_value);
        VecType right_selected = tmp2.template cast<DataType>();
        IdxType right_selected_index = utils::where<VecType, IdxType>(right_selected);


        MatType left_X = X(left_selected_index, Eigen::all);
        VecType left_y = y(left_selected_index);
        MatType right_X = X(right_selected_index, Eigen::all);
        VecType right_y = y(right_selected_index);

        if (left_X.rows() >= min_samples_leaf_) {
            std::shared_ptr<Node> left_child = std::make_shared<Node>();
            cur_node->is_leaf = false;
            cur_node->left_child = left_child;
            build_tree(left_X, left_y, left_child, cur_depth + 1);
        }

        if (right_X.rows() >= min_samples_leaf_) {
            std::shared_ptr<Node> right_child = std::make_shared<Node>();
            cur_node->is_leaf = false;
            cur_node->right_child = right_child;
            build_tree(right_X, right_y, right_child, cur_depth + 1);
        }
    }


    const DataType predict_label(const VecType& x, 
        std::shared_ptr<Node> cur_node) const{
        while (cur_node->left_child != nullptr) {
            if (x(cur_node->feature_index, 0) <= cur_node->feature_value) {
                cur_node = cur_node->left_child;
            }
            else {
                cur_node = cur_node->right_child;
            }
        }
        // VecType prob = cur_node->num_samples_per_class.array() / cur_node->num_samples_per_class.array().sum();
        // return prob;
        return cur_node->predict_value;
    }


public:
    DecisionTreeRegressor(): DecisionTree<DataType>(), 
        criterion_("squared_error"), 
        min_samples_split_(2), 
        min_samples_leaf_(0),
        max_depth_(4), 
        min_impurity_decrease_(1.0e-7),
        min_stdev_(1e-3) {};

    DecisionTreeRegressor(std::string criterion, 
        std::size_t min_samples_split, 
        std::size_t min_samples_leaf,
        std::size_t max_depth,
        double min_impurity_decrease,
        double min_stdev): DecisionTree<DataType>(), 
            criterion_(criterion), 
            min_samples_split_(min_samples_split), 
            min_samples_leaf_(min_samples_leaf),
            max_depth_(max_depth), 
            min_impurity_decrease_(min_impurity_decrease),
            min_stdev_(min_stdev) {};
    
     ~DecisionTreeRegressor() {
        delete_tree(root_);
    }
    
    /**
     * fit datatset
    */
     void fit(const MatType& X, 
        const VecType& y) {
        
        root_ = std::make_unique<Node>();
        build_tree(X, y, root_, 0);
    }

    const VecType predict(const MatType& X) const { 
        std::size_t num_samples = X.rows(), num_features = X.cols();
        VecType y_pred(num_samples);
        // y_pred.setZero();
        for (std::size_t i = 0; i < num_samples; ++i) {
            std::shared_ptr<Node> node = root_;
            auto row = X.row(i);
            auto p_i = predict_label(row.transpose(), node);
            y_pred(i, 0) = p_i;
        }
        return y_pred;
    }

};

} // decision_tree
} // openml

#endif /*METHODS_TREE_DECISION_TREE_REGRESSOR_HPP*/
