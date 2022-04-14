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
    
    struct Node {
        bool is_leaf;
        double impurity;
        std::size_t num_samples;
        std::size_t num_samples_per_class;
        std::size_t feature_index;
        DataType feature_value;
        DataType predict_class;
        std::unique_ptr<Node> left_child;
        std::unique_ptr<Node> right_child;

        Node():is_leaf(true),
            impurity(0.0),
            num_samples(0),
            num_samples_per_class(0),
            feature_index(ConstType<std::size_t>::max()),
            feature_value(ConstType<DataType>::quiet_NaN()),
            predict_class(ConstType<DataType>::quiet_NaN()),
            left_child(std::unique_ptr<Node>()), 
            right_child(std::unique_ptr<Node>()) {};

        explicit Node(bool is_leaf_,
            double impurity_ = 0.0,
            std::size_t num_samples_ = 0,
            std::size_t num_samples_per_class_ = 0, 
            std::size_t feature_index_ = ConstType<std::size_t>::max(), 
            DataType feature_value_ = ConstType<DataType>::quiet_NaN(), 
            DataType predict_class_ = ConstType<DataType>::quiet_NaN()): 
                is_leaf(is_leaf_),
                impurity(impurity_),
                num_samples(num_samples_), 
                num_samples_per_class(num_samples_per_class_),
                feature_index(feature_index_),
                feature_value(feature_value_),
                predict_class(predict_class_),
                left_child(std::unique_ptr<Node>()), 
                right_child(std::unique_ptr<Node>()){};

        ~Node(){};
    };
    using NodeType = Node;

    std::unique_ptr<NodeType> root;

    void delete_tree(std::unique_ptr<NodeType>& node) {
        if (node != nullptr) {
            delete_tree(node->left_child);
            delete_tree(node->right_child);
            std::move(node);
        }
    }


protected:
    std::size_t min_samples_split_;
    std::size_t min_samples_leaf_;
    std::size_t max_depth_; 
    double min_impurity_;


    /** pure virtual function to compute the inpurity */
    virtual const double compute_impurity (
        const VecType& y, 
        const VecType& left_y, 
        const VecType& right_y) const = 0;


    /**
     * choose a threshold from a given features vector, first we sort
     * this sample vector, then find the unique threshold
    */
    const VecType choose_feature_threshold(const VecType& x) const {
        std::size_t num_samples = x.rows();
        IdxType sorted_index = utils::argsort<VecType, IdxType>(x);
        VecType sorted_x = x(sorted_index);
        

        std::vector<DataType> stdvec_x;
        for (std::size_t i = 1; i < num_samples; ++i) {
            DataType mean = (sorted_x(i - 1) + sorted_x(i)) / 2;
            stdvec_x.push_back(mean);
        }

        std::sort(stdvec_x.begin(), stdvec_x.end());
        auto last = std::unique(stdvec_x.begin(), stdvec_x.end());
        stdvec_x.erase(last, stdvec_x.end());
        // VecType unique_x_values;
        Eigen::Map<VecType> unique_x(stdvec_x.data(), stdvec_x.size());

        return unique_x;
    }

    /**
     * Iterate through all unique values of feature column i and 
     * calculate the impurity. If this threshold resulted in a 
     * higher information gain than previously recorded save 
     * the threshold value and the feature index
    */
    const std::tuple<double, std::size_t, DataType> best_split(const MatType& X, 
            const VecType& y) const {
        
        std::size_t num_features = X.cols();
        double best_impurity = -ConstType<double>::infinity();
        std::size_t best_feature_index = ConstType<std::size_t>::max();
        DataType best_feature_value = ConstType<DataType>::quiet_NaN();

        for (std::size_t feature_index = 0; feature_index < num_features; ++feature_index) {
            VecType feature_values;
            feature_values = choose_feature_threshold(X.col(feature_index));
            for (auto& threshold : feature_values) {
                std::vector<std::size_t> left_index, right_index;
                std::tie(left_index, right_index) = data::divide_on_feature<MatType>(X, feature_index, threshold);

                VecType left_y, right_y;
                left_y = y(left_index);
                right_y = y(right_index);

                double impurity = this->compute_impurity(y, left_y, right_y);
                // std::cout << "impurity = " << impurity << std::endl;
                if (impurity > best_impurity) {
                    best_impurity = impurity;
                    best_feature_index = feature_index;
                    best_feature_value = threshold;
                }
            }
        }
        return std::make_tuple(best_impurity, best_feature_index, best_feature_value);
    }

    
    // void build_tree() {
        
    // }




public:
    DecisionTree(): min_samples_split_(2), 
        min_samples_leaf_(1),
        max_depth_(3), 
        min_impurity_(1e7) {};

    DecisionTree(std::size_t min_samples_split, 
        std::size_t min_samples_leaf,
        std::size_t max_depth,
        double min_impurity): 
            min_samples_split_(min_samples_split), 
            min_samples_leaf_(min_samples_leaf),
            max_depth_(max_depth), 
            min_impurity_(min_impurity) {};


    ~DecisionTree() {};





};

} // decision_tree
} // openml

#endif /*METHODS_TREE_BASE_HPP*/
