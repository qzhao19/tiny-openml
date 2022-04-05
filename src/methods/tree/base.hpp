#ifndef METHODS_TREE_BASE_HPP
#define METHODS_TREE_BASE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace decision_tree {


template<typename DataType>
class DecisionTree {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    
    
    std::size_t min_samples_split;
    std::size_t max_depth; 
    double min_impurity;

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
    using NodeType = Node;

    struct SplitRecords {
        /* data */
        double best_impurity;
        std::size_t best_feature_index;
        DataType best_feature_value; 
        SplitRecords(): best_impurity(0.0), 
            best_feature_index(ConstType<std::size_t>::max()),
            best_feature_value(ConstType<DataType>::quiet_NaN()),
        ~SplitRecords(){};
    };
    using SplitRecordsType = SplitRecords;
    
    std::unique_ptr<NodeType> root;

    void delete_tree(std::unique_ptr<NodeType>& node) {
        if (node != nullptr) {
            delete_tree(node->left_child);
            delete_tree(node->right_child);
            std::move(node);
        }
    }


protected:

    /** pure virtual function to compute the inpurity */
    virtual const double compute_impurity(
        const VecType& y, 
        const VecType& left_y, 
        const VecType& right_y) const = 0;





    VecType choose_split_feature(const VecType& x, const VecType& y) {
        std::size_t num_samples = y.rows();
        IdxType sorted_index = argsort<VecType, IdxType>(x);
        VecType sorted_x = x(sorted_index);
        VecType sorted_y = y(sorted_index);

        std::vector<Eigen::Index> split_index_vec;
        std::size_t count = 0;
        for (std::size_t i = 1; i < num_samples; ++i) {
            if (sorted_y(i - 1) != sorted_y(i)) {
                split_index_vec.push_back(i - 1);
                if (count > 1) {
                    split_index_vec.push_back(i);
                }
                ++count;
            }
            else {
                count = 0;
            }
        }
        // auto last = std::unique(split_index.begin(), split_index.end());
        // split_index.erase(last, v.end());

        Eigen::Map<IdxType> split_index(split_index_vec.data(), split_index_vec.size());
        return sorted_x(split_index);
    }




    const SplitRecordsType best_split(const MatType& X, const VecType& y) const {
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        MatType X_y(num_samples, num_features + 1);
        X_y = utils::hstack<MatType>(X, y);
                
        double best_impurity = 0.0;
        
        SplitRecordsType split_records;

        for (std::size_t feature_index = 0; feature_index < num_features; ++feature_index) {

            VecType feature_values = X_y.col(feature_index);
            VecType unique_values, unique_index;
            std::tie(unique_values, unique_index) = utils::unique<VecType>(feature_values);

            for (auto& threshold : unique_values) {
                MatType left_X_y, right_X_y;
                std::tie(left_X_y, right_X_y) = data::divide_on_feature<MatType>(X_y, feature_index, threshold);

                if (left_X_y.rows() >= min_samples_split && right_X_y.rows() >= min_samples_split) {

                    VecType left_y, right_y;
                    left_y = left_X_y.col(num_features);
                    right_y = right_X_y.col(num_features);

                    double impurity = this->compute_impurity(y, left_y, right_y);

                    if (impurity > best_impurity) {
                        best_impurity = impurity;

                        split_records.best_impurity = best_impurity;
                        split_records.best_feature_index = feature_index;
                        split_records.best_feature_value = threshold;

                    }
                }
            }
        }

        return split_records;

    }

    

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
