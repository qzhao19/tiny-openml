#ifndef METHODS_TREE_DECISION_TREE_CLASSIFIER_HPP
#define METHODS_TREE_DECISION_TREE_CLASSIFIER_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "base.hpp"
using namespace openml;

namespace openml {
namespace decision_tree {

template<typename DataType>
class DecisionTreeClassifier : public DecisionTree<DataType> {

private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    
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
    
    std::map<DataType, std::size_t> label_count_;
    std::size_t num_classes_;


    const double compute_impurity(const VecType& y, 
        const VecType& left_y, 
        const VecType& right_y) const {
        
        std::size_t num_samples = y.rows();
        std::size_t left_num_samples = left_y.rows();
        std::size_t right_num_samples = right_y.rows();

        double impurity = 0.0;
        if (criterion_ == "gini") {
            double left_gini = math::gini<VecType>(left_y);
            double right_gini = math::gini<VecType>(right_y);
            double g = static_cast<double>(left_num_samples) / static_cast<double>(num_samples) * left_gini +
                static_cast<double>(right_num_samples) / static_cast<double>(num_samples) * right_gini;
            impurity = 1.0 - g;
        }
        else if (criterion_ == "entropy") {
            double empirical_ent = math::entropy<VecType>(y);
            double left_ent = math::entropy<VecType>(left_y);
            double right_ent = math::entropy<VecType>(right_y);
            double ent = empirical_ent - 
                static_cast<double>(left_num_samples) / static_cast<double>(num_samples) * left_ent -
                    static_cast<double>(right_num_samples) / static_cast<double>(num_samples) * right_ent;
            impurity = ent;
        }
        else {
            throw std::invalid_argument("The criterion must be 'gini' or 'entropy'");
        }
        return impurity;
    }


    const std::tuple<VecType, DataType> compute_node_value(
        const VecType& y) const {
        
        std::map<DataType, std::size_t> label_count;
        for (std::size_t i = 0; i < y.rows(); i++) {
            label_count[y(i)]++;
        }

        // VecType value;
        VecType value_eigvec;
        std::vector<DataType> value_stdvec;
        for (auto& label : label_count_) {
            if (label_count.find(label.first) != label_count.end()) {
                value_stdvec.push_back(label_count.at(label.first));
            }
            else {
                value_stdvec.push_back(0);
            }
        }

        value_eigvec = common::vec2mat<VecType>(value_stdvec);
        auto value = common::argmax<MatType, VecType, IdxType>(value_eigvec, -1);
        return std::make_tuple(value_eigvec, value.value());
    }


    /** Build a decision tree by recursively finding the best split. */
    void build_tree(const MatType& X, 
        const VecType& y, 
        std::shared_ptr<Node>& cur_node, 
        std::size_t cur_depth) const {
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        
        DataType predict_value;
        VecType num_samples_per_class;
        
        std::tie(num_samples_per_class, predict_value) = compute_node_value(y);
        // std::cout << "num_samples_per_class = " << num_samples_per_class.transpose() << std::endl;

        cur_node->is_leaf = true;
        cur_node->num_samples = num_samples;
        cur_node->num_samples_per_class = num_samples_per_class;
        cur_node->predict_value = predict_value;
        
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
        IdxType left_selected_index = common::where<VecType, IdxType>(left_selected);

        auto tmp2 = (selected_x.array() > best_feature_value);
        VecType right_selected = tmp2.template cast<DataType>();
        IdxType right_selected_index = common::where<VecType, IdxType>(right_selected);

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

    /**
     * predict the label for each row data
     * @param x VecType of the data, row of input dataset
     * @param node TreeNode type, the tree 
    */
    const VecType predict_label_prob(const VecType& x, 
        std::shared_ptr<Node> cur_node) const{
        while (cur_node->left_child != nullptr) {
            if (x(cur_node->feature_index, 0) <= cur_node->feature_value) {
                cur_node = cur_node->left_child;
            }
            else {
                cur_node = cur_node->right_child;
            }
        }
        VecType prob = cur_node->num_samples_per_class.array() / cur_node->num_samples_per_class.array().sum();
        return prob;
    }


public:
    DecisionTreeClassifier(): DecisionTree<DataType>(), 
        criterion_("gini"), 
        min_samples_split_(2), 
        min_samples_leaf_(0),
        max_depth_(4), 
        min_impurity_decrease_(1.0e-7) {};

    DecisionTreeClassifier(std::string criterion,
        std::size_t min_samples_split, 
        std::size_t min_samples_leaf,
        std::size_t max_depth,
        double min_impurity_decrease): DecisionTree<DataType>(),
            criterion_(criterion),
            min_samples_split_(min_samples_split), 
            min_samples_leaf_(min_samples_leaf),
            max_depth_(max_depth), 
            min_impurity_decrease_(min_impurity_decrease) {};
    
    ~DecisionTreeClassifier() {
        delete_tree(root_);
    }

    /**
     * fit datatset
    */
     void fit(const MatType& X, 
        const VecType& y) {
        
        num_classes_ = 0;
        std::size_t num_samples = X.rows();
        for (std::size_t i = 0; i < num_samples; ++i) {
            label_count_[y(i)]++;
        }
        num_classes_ = label_count_.size();
        root_ = std::make_unique<Node>();
        build_tree(X, y, root_, 0);

    }

    /**
     * Return probability for the test data X.
     * @param X ndarray of shape (num_samples, num_features)
     *      input dataset
     * @return array-like of shape (num_samples, num_classes)
     *      Returns the probability of the samples for each class
     *      The columns correspond to the classes in sorted order
    */
    const MatType predict_prob(const MatType& X) const {
        std::size_t num_samples = X.rows(), num_features = X.cols();
        MatType prob(num_samples, num_classes_);
        prob.setOnes();
        for (std::size_t i = 0; i < num_samples; ++i) {
            std::shared_ptr<Node> node = root_;
            auto row = X.row(i);
            auto p_i = predict_label_prob(row.transpose(), node);
            prob.row(i) = p_i.transpose();
        }
        return prob;
    }

    const VecType predict(const MatType& X) const { 
        std::size_t num_samples = X.rows();
        MatType log_prob = predict_prob(X);
        auto y_pred_index = common::argmax<MatType, VecType, IdxType>(log_prob, 1);
        VecType y_pred_value(num_samples);
        
        std::size_t i = 0;
        for (auto& index : y_pred_index) {
            y_pred_value(i) = static_cast<DataType>(index);
            i++;
        }
        return y_pred_value;
    }

};

} // decision_tree
} // openml

#endif /*METHODS_TREE_DECISION_TREE_CLASSIFIER_HPP*/
