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
    
protected:
    std::string criterion_;
    
    const double compute_impurity(const VecType& y, 
        const VecType& left_y, 
        const VecType& right_y) const {
        
        std::size_t num_samples = y.rows();
        std::size_t left_num_samples = left_y.rows();
        std::size_t right_num_samples = right_y.rows();

        if (criterion_ == "gini") {
            double left_gini = math::gini<VecType>(left_y);
            double right_gini = math::gini<VecType>(right_y);
            double g = static_cast<double>(left_num_samples) / static_cast<double>(num_samples) * left_gini +
                static_cast<double>(right_num_samples) / static_cast<double>(num_samples) * right_gini;
            return 1.0 - g;
        }
        else if (criterion_ == "entropy") {
            double empirical_ent = math::entropy<VecType>(y);
            double left_ent = math::entropy<VecType>(left_y);
            double right_ent = math::entropy<VecType>(right_y);
            double ent = empirical_ent - 
                static_cast<double>(left_num_samples) / static_cast<double>(num_samples) * left_ent -
                    static_cast<double>(right_num_samples) / static_cast<double>(num_samples) * right_ent;
            return ent;
        }
    }

public:
    DecisionTreeClassifier(): DecisionTree<DataType>(), 
        criterion_("gini") {};


    explicit DecisionTreeClassifier(std::string criterion): 
        DecisionTree<DataType>(), 
        criterion_(criterion) {};

    DecisionTreeClassifier(std::string criterion,
        std::size_t min_samples_split, 
        std::size_t min_samples_leaf,
        std::size_t max_depth,
        double min_impurity): 
            DecisionTree<DataType>(
                min_samples_split, 
                min_samples_leaf,
                max_depth, 
                min_impurity),
            criterion_(criterion) {};

};

} // decision_tree
} // openml

#endif /*METHODS_TREE_DECISION_TREE_CLASSIFIER_HPP*/
