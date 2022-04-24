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
    
protected:
    std::string criterion_;
    double stdev_;
    
    const double compute_impurity(const VecType& y, 
        const VecType& left_y, 
        const VecType& right_y) const {
        
    }


public:
    DecisionTreeRegressor(): DecisionTree<DataType>(), 
        criterion_("gini"), 
        stdev_(1e-3) {};


    explicit DecisionTreeRegressor(std::string criterion, double stdev): 
        DecisionTree<DataType>(), 
        criterion_(criterion), 
        stdev_(stdev) {};

    DecisionTreeRegressor(std::string criterion,
        std::size_t min_samples_split, 
        std::size_t min_samples_leaf,
        std::size_t max_depth,
        double min_impurity, 
        double stdev): 
            DecisionTree<DataType>(
                min_samples_split, 
                min_samples_leaf,
                max_depth, 
                min_impurity),
            criterion_(criterion), 
            stdev_(stdev) {};

};

} // decision_tree
} // openml

#endif /*METHODS_TREE_DECISION_TREE_CLASSIFIER_HPP*/
